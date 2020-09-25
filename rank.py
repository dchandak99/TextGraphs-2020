from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
from tqdm.notebook import tqdm
from tqdm import trange
import pickle
import nltk
import math
import os
import random
import re
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, 
                                  BartConfig, BartTokenizer, BartForSequenceClassification,
                          LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer,
                          AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                          ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer,
                          ReformerConfig, ReformerForSequenceClassification, ReformerTokenizer,
                          MobileBertConfig, MobileBertForSequenceClassification, MobileBertTokenizer,
                          AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          )
import sys
import warnings
from collections import namedtuple, OrderedDict
from functools import partial
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.validation import check_is_fitted
from pathlib import Path
from tqdm import tqdm
import pandas as po
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from functools import partial
from multiprocessing import Pool
from scipy import sparse
from sklearn import pipeline, feature_extraction, metrics

class ListShouldBeEmptyWarning(UserWarning):
  pass

Question = namedtuple('Question', 'id explanations')
Explanation = namedtuple('Explanation', 'id role')

def format_role(role: str) -> str:
  if role == "NE":
    return "NEG"
  return role

def load_gold(filepath_or_buffer, sep='\t'):
  df = po.read_csv(filepath_or_buffer, sep=sep)
  gold = OrderedDict()
  for _, row in df[['QuestionID', 'explanation']].dropna().iterrows():
    # explanations = OrderedDict((uid.lower(), Explanation(uid.lower(), role))
    explanations = OrderedDict(
        (uid.lower(), Explanation(uid.lower(), format_role(role)))
        for e in row['explanation'].split()
        for uid, role in (e.split('|', 1), ))
    question = Question(row['QuestionID'].lower(), explanations)
    gold[question.id] = question
  return gold

def load_pred(filepath_or_buffer, sep='\t'):
  df = po.read_csv(filepath_or_buffer, sep=sep, names=('question', 'explanation'))
  pred = OrderedDict()
  for question_id, df_explanations in df.groupby('question'):
    pred[question_id.lower()] = list(OrderedDict.fromkeys(df_explanations['explanation'].str.lower()))
  return pred

# From https://github.com/arosh/BM25Transformer/blob/master/bm25.py
class BM25Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, use_idf=True, k1=2.0, b=0.75):
    self.use_idf = use_idf
    self.k1 = k1
    self.b = b

  def fit(self, X):
    if not sp.issparse(X):
      X = sp.csc_matrix(X)
    if self.use_idf:
      n_samples, n_features = X.shape
      df = _document_frequency(X)
      idf = np.log((n_samples - df + 0.5) / (df + 0.5))
      self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
    return self

  def transform(self, X, copy=True):
    if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
      X = sp.csr_matrix(X, copy=copy)
    else:
      X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

    n_samples, n_features = X.shape
    dl = X.sum(axis=1)
    sz = X.indptr[1:] - X.indptr[0:-1]
    rep = np.repeat(np.asarray(dl), sz)
    avgdl = np.average(dl)
    data = X.data*(self.k1+1)/(X.data+self.k1*(1 - self.b + self.b * rep / avgdl))
    X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    if self.use_idf:
      check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')
      expected_n_features = self._idf_diag.shape[0]
      if n_features != expected_n_features:
        raise ValueError("Input has n_features=%d while the model"
                         " has been trained with n_features=%d" %
                         (n_features, expected_n_features))
      X = X * self._idf_diag
    return X

class MyBM25Transformer(BM25Transformer):
  def fit(self, x, y=None):
    super().fit(x)


class BM25Vectorizer(feature_extraction.text.TfidfVectorizer):
  def __init__(self):
    self.vec = pipeline.make_pipeline(feature_extraction.text.CountVectorizer(binary=True),MyBM25Transformer(),)
    super().__init__()
  def fit(self, raw_documents, y=None):
    return self.vec.fit(raw_documents)
  def transform(self, raw_documents, copy=True):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=FutureWarning)
      return self.vec.transform(raw_documents)

def maybe_concat_texts(texts: list) -> list:
  if type(texts[0]) == str:
    return texts
  elif type(texts[0]) == list:
    return [" ".join(sublst) for sublst in texts]
  else:
    raise TypeError(f"Unknown data type: {type(texts[0])}")

def repeat(seed: list, idx2nns: dict) -> dict:
  ranking = {item: idx for idx, item in enumerate(seed)}
  idx2nns = dict(idx2nns)
  while True:
    old_ranking = dict(ranking)
    for idx in old_ranking.keys():
      while True:
        if len(idx2nns[idx]) == 0:
          break
        n = idx2nns[idx].pop(0)
        if n not in ranking:
          ranking[n] = len(ranking)
          break
    if len(ranking) == len(old_ranking):
      break
  scores = {k: -v for k, v in ranking.items()}
  return scores

def recurse(seed: list,scores: dict,idx2nns: dict,iteration: int,n: int = 100, max_iter: int = 2,) -> None:
  if iteration < max_iter:
    for idx in seed[:n]:
      if idx in scores:
        scores[idx] += 1
      else:
        scores[idx] = 1
    for idx in seed[:n]:
      if scores[idx] > 1:
        continue
      new_seed = idx2nns[idx][1:]  
      recurse(new_seed, scores, idx2nns, iteration + 1, n, max_iter)

def simple_nn_ranking(df: po.DataFrame,df_exp: po.DataFrame,idx: int,annoy_index=None,) -> list:
  return annoy_index.get_nns_by_vector(df.embedding.iloc[idx], n=len(df_exp))

def recurse_concat(x: sparse.csr_matrix,vectors_e: sparse.csr_matrix,seq: list,maxlen: int = 128,top_n: int = 1,scale: float = 1.25,idx2idx_canon: dict = None,) -> None:
  if len(seq) < maxlen:
    matrix_dist = metrics.pairwise.cosine_distances(x, vectors_e)
    ranks = [np.argsort(distances) for distances in matrix_dist]
    assert len(ranks) == 1
    rank = ranks[0]
    seen = set(seq)
    count = 0
    for idx in rank:
      if count == top_n:
        break
      idx_canon = idx if idx2idx_canon is None else idx2idx_canon[idx]
      if idx_canon not in seen:
        e = vectors_e[idx] / (scale**len(seq))
        new = x.maximum(e)
        seq.append(idx_canon)
        count += 1
        recurse_concat(new, vectors_e, seq)

def get_idx2idx_canon(df_exp: po.DataFrame) -> dict:
  uids_canon = df_exp.uid.apply(remove_combo_suffix).tolist()
  idx2uid_canon = {idx: uid for idx, uid in enumerate(uids_canon)}
  e_canon = df_exp[df_exp.uid.isin(set(uids_canon))]
  print("Num canonical explanations:", len(e_canon))
  uid2idx_canon = {uid: idx for idx, uid in zip(e_canon.index, e_canon.uid)}
  return {
      idx: uid2idx_canon[idx2uid_canon[idx]]
      for idx in idx2uid_canon.keys()
  }


def recurse_concat_helper(x: sparse.csr_matrix,vectors_e: sparse.csr_matrix,idx2idx_canon: dict,) -> list:
  seq = []
  recurse_concat(x, vectors_e, seq, idx2idx_canon=idx2idx_canon)
  return seq

def get_recurse_concat_ranks(df: po.DataFrame, df_exp: po.DataFrame) -> list:
  field_q = field_e = "lemmas"
  q = maybe_concat_texts(df[field_q].tolist())
  e = maybe_concat_texts(df_exp[field_e].tolist())
  vec = BM25Vectorizer()
  vec.fit(q + e)
  vectors_q = vec.transform(q)
  vectors_e = vec.transform(e)
  idx2idx_canon = get_idx2idx_canon(df_exp)
  assert len(df) == len(q) == vectors_q.shape[0]
  _recurse_concat_helper = partial(recurse_concat_helper,vectors_e=vectors_e,idx2idx_canon=idx2idx_canon,)
  with Pool() as pool:
    xs = [vectors_q[i] for i in range(len(df))]
    results = pool.imap(_recurse_concat_helper, xs, chunksize=10)
    ranks = list(tqdm(results, total=len(xs)))
  return ranks

def get_tfidf_ranking(df: po.DataFrame,df_exp: po.DataFrame,field_q: str = "q_reformat", field_e: str = "text",mode: str = "tfidf",) -> list:
  q = maybe_concat_texts(df[field_q].tolist())
  e = maybe_concat_texts(df_exp[field_e].tolist())
  vec = {
      "tfidf": feature_extraction.text.TfidfVectorizer(binary=True, norm='l2'),
  }[mode]
  vec.fit(q + e)
  vectors_q = vec.transform(q)
  vectors_e = vec.transform(e)
  matrix_dist = metrics.pairwise.cosine_distances(vectors_q, vectors_e)
  return [np.argsort(distances) for distances in matrix_dist]

def remove_duplicates(lst: list) -> list:
  seen = set()
  new = []
  for item in lst:
    if item not in seen:
      new.append(item)
      seen.add(item)
  return new

def add_missing_idxs(old, idxs_sample):
  old = remove_duplicates(old)
  set_old = set(old)
  set_all = set(idxs_sample)
  missing = list(set_all - set_old)
  np.random.shuffle(missing)
  new = list(old)
  new.extend(missing)
  assert len(new) == len(set_all), (len(new), len(set_all), len(missing))
  assert all([a == b for a, b in zip(new[:len(old)], old)])
  assert set(new) == set_all
  return new

def format_predict_line(question_id, explanation_uid):
  return question_id + "\t" + explanation_uid

def remove_combo_suffix(explanation_uid):
  return explanation_uid.split("_")[0]

def deduplicate_combos(ranks, df_exp):
  idx2idx_canon = get_idx2idx_canon(df_exp)
  def process(lst):
    return remove_duplicates([idx2idx_canon[idx] for idx in lst])
  return [process(lst) for lst in ranks]

def ideal_rerank(ranks, df, df_exp, top_n=64):
  def get_combo_idxs(_df_exp):
    uid2idxs = {}
    uids = _df_exp.uid.apply(remove_combo_suffix).tolist()
    for _idx, uid in enumerate(uids):
      if uid not in uid2idxs:
        uid2idxs[uid] = []
      uid2idxs[uid].append(_idx)
    return {_idx: uid2idxs[uid] for _idx, uid in enumerate(uids)}
  idxs_gold = df.exp_idxs.tolist()
  lengths_front, lengths_gold = [], []
  assert len(ranks) == len(idxs_gold) == len(df)
  idx2combo_idxs = get_combo_idxs(df_exp)
  for i in range(len(ranks)):
    temp = []
    for idx in idxs_gold[i]:
      temp.extend(idx2combo_idxs[idx])
    idxs_gold[i] = list(set(temp))
    if len(idxs_gold[i]) > 0:  
      front = [idx for idx in ranks[i][:top_n] if idx in idxs_gold[i][:top_n]]
      back = [idx for idx in ranks[i] if idx not in front]
      new = front + back
      assert set(new) == set(ranks[i])
      ranks[i] = new
      lengths_front.append(len(front))
      lengths_gold.append(len(idxs_gold[i]))
  if len(lengths_front) > 0:
    def _process(lengths):
      return round(sum(lengths) / len(lengths), 3)
    f = _process(lengths_front)
    g = _process(lengths_gold)
    print(f"\nGold present in rankings | Predicted: {f} | Total: {g}")
  return ranks

def augment_ranks(ranks_a, ranks_b):
  assert len(ranks_a) == len(ranks_b)
  augmented = []
  for front, back in zip(ranks_a, ranks_b):
    set_front = set(front)
    back = [r for r in back if r not in set_front]
    augmented.append(front + back)
  return augmented

def pred_emb_nn_ranking(df: po.DataFrame,idx: int,model=None,annoy_index=None,df_exp: po.DataFrame = None,) -> list:
  inp = [df.embedding.iloc[idx]]
  out = model.predict(inp)
  pred_emb = out[0]
  return annoy_index.get_nns_by_vector(pred_emb, n=len(df_exp))

def get_ranks(df: po.DataFrame,df_exp: po.DataFrame, mode: str = "tfidf", use_embed: bool = False, use_recursive_tfidf: bool = None, field_q: str = "lemmas", field_e: str = "lemmas",) -> list:
  tfidf_ranking = get_tfidf_ranking(df, df_exp, field_q, field_e)
  tfidf_ranking = deduplicate_combos(tfidf_ranking, df_exp)
  if use_recursive_tfidf:
    recurse_ranks = get_recurse_concat_ranks(df, df_exp)
    recurse_ranks = deduplicate_combos(recurse_ranks, df_exp)
    tfidf_ranking = augment_ranks(recurse_ranks, tfidf_ranking)
  if use_embed:
    idx2nns = {
        idx: list(df_exp.nn_exp.iloc[idx])
        for idx in range(len(df_exp))
    }
  else:
    idx2nns = None
  def scores2ranks(_scores):
    return sorted(list(_scores.keys()),key=lambda _i: _scores[_i],reverse=True,)
  uids_all = df_exp.uid.apply(remove_combo_suffix)
  orig_exp_set = set(uids_all.tolist())
  orig_exp_idxs = df_exp[df_exp.uid.isin(orig_exp_set)].index.tolist()
  has_missing = False
  ranks = []

  for i in tqdm(range(len(df))):
    if mode == "repeat":
      nearest = df.nn_exp.iloc[i]
      scores = repeat(nearest, idx2nns)
      ranked_idxs = scores2ranks(scores)

    elif mode == "recurse":
      nearest = df.nn_exp.iloc[i]
      scores = {}
      recurse(seed=nearest, scores=scores, idx2nns=idx2nns, iteration=0)
      ranked_idxs = scores2ranks(scores)

    elif mode == "simple_nn":
      ranked_idxs = simple_nn_ranking(df, df_exp, i)

    elif mode == "pred_emb":
      ranked_idxs = pred_emb_nn_ranking(df, i)

    elif mode == "tfidf":
      ranked_idxs = list(tfidf_ranking[i])

    else:
      raise Exception(f"Unknown mode: {mode}")

    if len(ranked_idxs) != len(orig_exp_set):
      if not has_missing:
        print("Filling in missing rankings with random order")
        has_missing = True
      ranked_idxs = add_missing_idxs(ranked_idxs, idxs_sample=orig_exp_idxs)

    ranks.append(ranked_idxs)
  return ranks


def get_preds(ranks: list, df: po.DataFrame, df_exp: po.DataFrame) -> list:
  preds = []
  assert len(ranks) == len(df)
  uids = df_exp.uid.apply(remove_combo_suffix).values
  qids = df.QuestionID.tolist()
  for i in range(len(df)):
    uids_pred = uids[ranks[i]]
    preds.extend([format_predict_line(qids[i], uid) for uid in uids_pred])
  return preds
  
def write_preds(preds: list, path: str = "predict.txt") -> None:
  with open(path, "w") as f:
    f.write("\n".join(preds))

def test_write_preds():
  preds = [
      "VASoL_2008_3_26\t14de-6699-6b2e-a5d1",
      "VASoL_2008_3_26\t14de-6699-6b2e-a5d1",
  ]
  path = "temp.txt"
  write_preds(preds, path)
  with open(path) as f:
    for line in f:
      print(repr(line))
    




