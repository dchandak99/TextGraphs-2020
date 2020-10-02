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
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
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
from paths import get_path_predict, get_path_df_scores, get_path_q
from rank import maybe_concat_texts, BM25Vectorizer, deduplicate_combos, remove_combo_suffix, add_missing_idxs, get_ranks, get_preds


SEP = "#" * 100 + "\n"
MODE_TRAIN = "train"
MODE_DEV = "dev"
MODE_TEST = "test"

def obtain_useful(path_data, path_tables, mode='train',top_k = 500):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_exp = get_df_explanations(path_tables, path_data)
    uid2idx = {uid: idx for idx, uid in enumerate(df_exp.uid.tolist())}
    uids = df_exp.uid.apply(remove_combo_suffix).values 
    path_q = get_path_q(path_data, mode)
    df = get_questions(str(path_q), uid2idx, path_data)
    ranks = get_ranks(df, df_exp, use_embed=False, use_recursive_tfidf=True)
    preds = get_preds(ranks, df, df_exp)
    df_exp_copy = df_exp.set_index('uid')
    uid2text = df_exp_copy['text'].to_dict()
    return df, df_exp, uids, uid2idx, uid2text, ranks, preds
    
def obtain_model_names_and_classes(model_name='roberta', model_type='roberta-base'):
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'bart': (BartConfig, BartForSequenceClassification, BartTokenizer),
        'longformer':(LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer),
        'albert':(AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        'electra':(ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
        'reformer':(ReformerConfig, ReformerForSequenceClassification, ReformerTokenizer),
        'distilbert':(DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        'scibert':( AutoModel, AutoModelForSequenceClassification,AutoTokenizer),    
    }
    types_of_models=[{'bert':['bert-base-uncased']},{'xlm':['xlm-mlm-en-2048']},{'roberta':['roberta-base']},{'bart':["facebook/bart-base"]},{'longformer':['allenai/longformer-base-4096']},{'albert':['albert-xlarge-v2','albert-large-v2','albert-base-v2']},{'electra':['google/electra-large-generator']},{'reformer':['google/reformer-crime-and-punishment','google/reformer-enwik8']},{'distilbert':['distilbert-base-uncased']},{'scibert':['allenai/scibert_scivocab_uncased']}]
    print("Choose from the list of models and their respective pretrained versions:")
    print(types_of_models)
    model_with_no_token_types =['roberta', 'bart' ,'longformer','albert','electra','reformer','distilbert','scibert']
    config_class, model_classifier, model_tokenizer = MODEL_CLASSES[model_name]
    tokenizer = model_tokenizer.from_pretrained(model_type)
    return MODEL_CLASSES, model_with_no_token_types, tokenizer

def compute_ranks(true, pred):
  ranks = []
  if not true or not pred:
    return ranks
  targets = list(true)
  for i, pred_id in enumerate(pred):
    for true_id in targets:
      if pred_id == true_id:
        ranks.append(i + 1)
        targets.remove(pred_id)
        break
  if targets:
    warnings.warn(
        'targets list should be empty, but it contains: ' + ', '.join(targets),
        ListShouldBeEmptyWarning)
    for _ in targets:
      ranks.append(0)
  return ranks
  
def average_precision_score(gold, pred):
    if not gold or not pred:
        return 0.
    correct = 0
    ap = 0.
    true = set(gold)
    for rank, element in enumerate(pred):
        if element in true:
            correct += 1
            ap += correct / (rank + 1.)
            true.remove(element)
    return ap / len(gold)

def prepare_features(seq_1,seq_2, max_seq_length = 300, 
             zero_pad = True, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)
    tokens_b = tokenizer.tokenize(seq_2)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    for token in tokens_b:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask 
    input_mask = [1] * len(input_ids)
    ##Segment_ids
    segment_ids = [0]*(len(tokens_a)+1)
    segment_ids+= [1]*(len(tokens_b)+1)
    segment_ids = [0] + segment_ids

    ## Zero-pad sequence length
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
    #return torch.tensor(input_ids).unsqueeze(0), input_mask
    return input_ids, input_mask ,segment_ids


class DefaultLemmatizer:
	"""
	Works best to transform texts before and also get lemmas during tokenization
	"""
	def __init__(self, path_data: Path = None) -> None:
		if path_data is None:
			self.word2lemma = {}
		else:
			path_anno = path_data.joinpath("annotation")
			path = path_anno.joinpath("lemmatization-en.txt")

			def read_csv(_path: str, names: list = None) -> po.DataFrame:
				return po.read_csv(_path, header=None, sep="\t", names=names)

			df = read_csv(str(path), ["lemma", "word"])
			#path_extra = path_anno.joinpath(
			#		"expl-tablestore-export-2017-08-25-230344/tables/LemmatizerAdditions.tsv"
			#)
			path_extra = path_anno.joinpath("LemmatizerAdditions.tsv")
			df_extra = read_csv(str(path_extra), ["lemma", "word", "useless"])
			df_extra.drop(columns=["useless"], inplace=True)
			df_extra.dropna(inplace=True)

			length_old = len(df)
			# df = po.concat([df, df_extra])  # Actually concat extra hurts MAP (0.462->0.456)
			print(
					f"Default lemmatizer ({length_old}) concatenated (or not) with extras ({len(df_extra)}) -> {len(df)}"
			)

			lemmas = df.lemma.tolist()
			words = df.word.tolist()

			def only_alpha(text: str) -> str:
				# Remove punct eg dry-clean -> dryclean so
				# they won't get split by downstream tokenizers
				return "".join([c for c in text if c.isalpha()])

			self.word2lemma = {
					words[i].lower(): only_alpha(lemmas[i]).lower()
					for i in range(len(words))
			}

	def transform(self, raw_texts: list) -> list:
		def _transform(text: str):
			return " ".join(
					[self.word2lemma.get(word) or word for word in text.split()])

		return [_transform(text) for text in raw_texts]

# Basic function from tfidf_baseline
def read_explanations(path):
  header = []
  uid = None

  df = po.read_csv(path, sep='\t')

  for name in df.columns:
    if name.startswith('[SKIP]'):
      if 'UID' in name and not uid:
        uid = name
    else:
      header.append(name)

  if not uid or len(df) == 0:
    warnings.warn('Possibly misformatted file: ' + path)
    return []

  return df.apply(
      lambda r:
      (r[uid], ' '.join(str(s) for s in list(r[header]) if not po.isna(s))),
      1).tolist()

# Returns Tokens and Lemmas 
def preprocess_texts( 
		texts: list,
		path_data: Path = None,
) -> (list, list):
	# NLTK tokenizer on par with spacy and less complicated
	tokenizer = nltk.tokenize.TreebankWordTokenizer()
	default_lemmatizer = DefaultLemmatizer(path_data)
	# wordnet_lemmatizer doesn't help
	texts = default_lemmatizer.transform(texts)
	stops = set(nltk.corpus.stopwords.words("english"))

	def lemmatize(token):
		return default_lemmatizer.word2lemma.get(token) or token

	def process(
			text: str,
			_tokenizer: nltk.tokenize.TreebankWordTokenizer,
	) -> (list, list):
		_tokens = _tokenizer.tokenize(text.lower())
		_lemmas = [
				lemmatize(_tok) for _tok in _tokens
				if _tok not in stops and not _tok.isspace()
		]
		return _tokens, _lemmas

	tokens, lemmas = zip(*[process(text, tokenizer) for text in tqdm(texts)])
	return tokens, lemmas

def exp_skip_dep(path_exp: Path, col: str = "[SKIP] DEP", save_temp: bool = True,) -> str:
    """
    Remove rows that have entries in deprecated column
    according to https://github.com/umanlp/tg2019task/issues/2
    """
    try:
        df = po.read_csv(path_exp, sep="\t")
    except:
        print(path_exp)
        raise
    if col in df.columns:
        df = df[df[col].isna()]
    path_new = "temp.tsv" if save_temp else Path(path_exp).name
    df.to_csv(path_new, sep="\t", index=False)
    return path_new

def get_questions(
		path: str,
		uid2idx: dict = None,
		path_data: Path = None,
) -> po.DataFrame:
	"""
	Identify correct answer text and filter out wrong distractors from question string
	Get tokens and lemmas
	Get explanation sentence ids and roles
	"""
	# Dropping questions without explanations hurts score
	df = po.read_csv(path, sep="\t")
	df = add_q_reformat(df)

	# Preprocess texts
	tokens, lemmas = preprocess_texts(df.q_reformat.tolist(), path_data)
	df["tokens"], df["lemmas"], df["embedding"] = tokens, lemmas, None

	# Get explanation uids and roles
	exp_uids = []
	exp_roles = []
	exp_idxs = []
	for exp_string in df.explanation.values:
		_uids, _roles = extract_explanation(exp_string)
		uids = []
		roles = []
		idxs = []
		assert len(_uids) == len(_roles)
		for i in range(len(_uids)):
			if _uids[i] not in uid2idx:
				continue
			uids.append(_uids[i])
			roles.append(_roles[i])
			idxs.append(uid2idx[_uids[i]])
		exp_uids.append(uids)
		exp_roles.append(roles)
		exp_idxs.append(idxs)
	df["exp_uids"], df["exp_roles"], df[
			"exp_idxs"] = exp_uids, exp_roles, exp_idxs

	print(df.shape)
	return df

def get_df_explanations(
		path_tables: str,
		path_data: Path = None,
):
	"""
	Make a dataframe of explanation sentences (~5000)
	"""
	explanations = []
	columns = None
	for p in Path(path_tables).iterdir():
		columns = ["uid", "text"]
		p = exp_skip_dep(p)
		# p = save_unique_phrases(Path(p))
		explanations += read_explanations(str(p))
	df = po.DataFrame(explanations, columns=columns)
	df = df.drop_duplicates("uid").reset_index(drop=True)  # 3 duplicate uids
	tokens, lemmas = preprocess_texts(df.text.tolist(), path_data=path_data)
	df["tokens"], df["lemmas"], df["embedding"] = tokens, lemmas, None
	print("Explanation df shape:", df.shape)
	return df

def extract_explanation(exp_string):
	"""
	Convert raw string (eg "uid1|role1 uid2|role2" -> [uid1, uid2], [role1, role2])
	"""
	if type(exp_string) != str:
		return [], []
	uids = []
	roles = []
	for uid_and_role in exp_string.split():
		uid, role = uid_and_role.split("|")
		uids.append(uid)
		roles.append(role)
	return uids, roles

def add_q_reformat(df: po.DataFrame) -> po.DataFrame:
	q_reformat = []
	questions = df.question.values
	answers = df["AnswerKey"].values
	char2idx = {char: idx for idx, char in enumerate(list("ABCDE"))}

	#print(answers)
	#print(char2idx)

	for i in range(len(df)):
		q, *options = split_question(questions[i])
		try:
			if answers[i] in ['A', 'B', 'C', 'D', 'E']:
				idx_option = char2idx[answers[i]]
			elif answers[i] in ['1', '2', '3', '4', '5']:
				idx_option = int(answers[i]) - 1
			else:
				print(answers[i])
				print(type(answers[i]))
				raise ValueError
		except:
			print(answers[i])
			raise
		try:
			q_reformat.append(" ".join([q.strip(), options[idx_option].strip()]))
		except:
			print(idx_option)
			print(options)
			raise
	df["q_reformat"] = q_reformat
	return df


def split_question(q_string):
	"""
	Split on option parentheses (eg "Question (A) option1 (B) option2" -> [Question, option 1, option2])
	Note that some questions have more or less than 4 options
	"""
	return re.compile("\\(.\\)").split(q_string)


def preproc_trn_data(df: po.DataFrame) -> po.DataFrame:
  """
  Three reasons to remove qe pairs with score == 1.0:
  1. Questions without explanations will always result in 1.0
  2. Valid qe pairs with 1.0 means the explanation is completely unrelated
      which is too easy for the model
  3. They skew the label/label distribution
  """
  print("Preprocessing train bert data (df_scores)")
  old_length = len(df)
  df = df[~(df.score == 1.0)]
  print(f"Dropping irrelevant explanations ({old_length} -> {len(df)})")
  df = df.sample(frac=1).reset_index(drop=True)  # shuffle
  print("Plotting histrogram distribution of scores")
  df.score.hist(bins=50)
  return df


def make_score_data(
    df: po.DataFrame,
    df_exp: po.DataFrame,
    rankings: list,
    top_n: int = 64,
) -> po.DataFrame:
  q = maybe_concat_texts(df["lemmas"].tolist())
  e = maybe_concat_texts(df_exp["lemmas"].tolist())
  vec = BM25Vectorizer()
  vec.fit(q + e)
  vectors_e = vec.transform(e)

  # Gold explanations
  def concat_exp_text(exp_idxs):
    def concat_lst(lst):
      return " ".join(lst)

    return " ".join(df_exp.lemmas.iloc[exp_idxs].apply(concat_lst).tolist())

  e_gold = df.exp_idxs.apply(concat_exp_text)
  vectors_e_gold = vec.transform(e_gold)
  matrix_dist_gold = cosine_distances(vectors_e_gold, vectors_e)
  top_ranks = [ranks[:top_n] for ranks in rankings]
  top_dists = [
      matrix_dist_gold[i][top_ranks[i]] for i in range(len(top_ranks))
  ]

  data = []
  for i in range(len(top_ranks)):
    text_q = df.q_reformat.iloc[i]
    texts_e = df_exp.text.iloc[top_ranks[i]].tolist()
    for j in range(top_n):
      data.append([text_q, texts_e[j], top_dists[i][j]])

  df_scores = po.DataFrame(data, columns=["text_q", "text_e", "score"])
  print(df_scores.shape)
  return df_scores

def prepare_rerank_data(df: po.DataFrame, df_exp: po.DataFrame, ranks: list,
                        mode: str) -> None:
  path_df_scores = get_path_df_scores(mode)

  if mode == MODE_TRAIN:
    df_scores = make_score_data(df, df_exp, ranks)
    df_scores.to_csv(get_path_df_scores(mode, clean_trn=True), index=False)
    df_scores = preproc_trn_data(df_scores)
  else:
    # df_scores = make_score_data(df, df_exp, ranks, top_n=1024)
    df_scores = make_score_data(df, df_exp, ranks)
  print(SEP, "Preparing rerank data")
  print("Saving rerank data to:", path_df_scores)
  df_scores.to_csv(path_df_scores, index=False)
