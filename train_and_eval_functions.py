from pathlib import Path
from tqdm.notebook import tqdm
from tqdm import trange
import pandas as po
import numpy as np
import warnings
import pickle
import nltk
import math
import os
import random
import re
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
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
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)
from paths import get_path_q, get_path_df_scores, get_path_predict
from utils import get_df_explanations, get_questions, average_precision_score
from rank import get_ranks, get_preds, ideal_rerank, remove_combo_suffix, format_predict_line, write_preds
 
SEP = "#" * 100 + "\n"
path_data   = Path("data/")
path_tables = path_data.joinpath("raw/tables")
device = 'cuda'

def make_dataset_test(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, top_k, model_with_no_token_types, model_name='roberta'):
  all_input_ids = []
  all_token_type_ids = []
  all_attention_masks = []
  all_labels = [] 
  rerank = []

  for i in tqdm(range(len(df))):
      uids_pred = uids[ranks[i]]
          
      question = df.iloc[i]['q_reformat']
      pred = [uid2text[j] for j in uids_pred][:top_k]                    
      gold = df_exp.loc[df_exp['uid'].isin(df.iloc[i]['exp_uids'])]['text'].tolist()

      labels = [1 if i in gold else 0 for i in pred]

      row = {}
      row['ques'] = question
      row['pred'] = pred
      row['gold'] = gold
      
      rerank.append(row)

      for i, p in enumerate(pred):
        label = labels[i]

        if model_name in model_with_no_token_types:
            encoded_input   = tokenizer(question, p, padding='max_length', max_length=100, truncation='longest_first', return_tensors="pt")
            input_ids       = encoded_input['input_ids'].tolist()
            attention_mask  = encoded_input['attention_mask'].tolist()

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(label)
            
        elif model_name=='bert':
          encoded_input   = tokenizer(question, p, padding='max_length', max_length=100, truncation='longest_first', return_tensors="pt")
          input_ids       = encoded_input['input_ids'].tolist()
          token_type_ids  = encoded_input['token_type_ids'].tolist()
          attention_mask  = encoded_input['attention_mask'].tolist()

          all_input_ids.append(input_ids)
          all_token_type_ids.append(token_type_ids)
          all_attention_masks.append(attention_mask)
          all_labels.append(label)  

  if model_name in model_with_no_token_types:
    all_input_ids = torch.tensor(all_input_ids).squeeze()
    #all_token_type_ids = torch.tensor(all_token_type_ids).squeeze()
    all_attention_masks = torch.tensor(all_attention_masks).squeeze()
    all_labels = torch.tensor(all_labels) 
    dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels)
  
  elif model_name=='bert':
    all_input_ids = torch.tensor(all_input_ids).squeeze()
    all_token_type_ids = torch.tensor(all_token_type_ids).squeeze()
    all_attention_masks = torch.tensor(all_attention_masks).squeeze()
    all_labels = torch.tensor(all_labels)    
    dataset = TensorDataset(all_input_ids,all_token_type_ids, all_attention_masks, all_labels)
  return dataset, rerank

def make_dataset_train(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, top_k, model_with_no_token_types, model_name='roberta'):
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    all_labels = [] 
    rerank = []
    for i in tqdm(range(len(df))):
        uids_pred = uids[ranks[i]]
        question = df.iloc[i]['q_reformat']
        pred_imb = [uid2text[j] for j in uids_pred][:top_k]                    
        gold = df_exp.loc[df_exp['uid'].isin(df.iloc[i]['exp_uids'])]['text'].tolist()
        pred_0 = [p for p in pred_imb if p not in gold]
        pred_1 = [p for p in pred_imb if p in gold]

        if len(pred_1) == 0:
          continue

        pred = []
        pred += (pred_1*(int((top_k/2)/len(pred_1))+1))[:int(top_k/2)]
        pred += pred_0[:int(top_k/2)]
        pred = random.sample(pred, k=len(pred))

        labels = [1 if i in gold else 0 for i in pred]

        row = {}
        row['ques'] = question
        row['pred'] = pred
        row['gold'] = gold
        rerank.append(row)

        for i, p in enumerate(pred):
          label = labels[i]

          if model_name in model_with_no_token_types:
            encoded_input   = tokenizer(question, p, padding='max_length', max_length=100, truncation='longest_first', return_tensors="pt")
            input_ids       = encoded_input['input_ids'].tolist()
            attention_mask  = encoded_input['attention_mask'].tolist()
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(label)
            
          else:
            encoded_input   = tokenizer(question, p, padding='max_length', max_length=100, truncation='longest_first', return_tensors="pt")
            input_ids       = encoded_input['input_ids'].tolist()
            token_type_ids  = encoded_input['token_type_ids'].tolist()
            attention_mask  = encoded_input['attention_mask'].tolist()
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(label)            

    if model_name in model_with_no_token_types:
      all_input_ids = torch.tensor(all_input_ids).squeeze()
      all_attention_masks = torch.tensor(all_attention_masks).squeeze()
      all_labels = torch.tensor(all_labels) 
      dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels)

    else:
      all_input_ids = torch.tensor(all_input_ids).squeeze()
      all_token_type_ids = torch.tensor(all_token_type_ids).squeeze()
      all_attention_masks = torch.tensor(all_attention_masks).squeeze()
      all_labels = torch.tensor(all_labels)    
      dataset = TensorDataset(all_input_ids,all_token_type_ids, all_attention_masks, all_labels)

    return dataset, rerank
    
def train_model(df, df_exp, uids, uid2idx, uid2text, ranks, preds, MODEL_CLASSES, model_with_no_token_types, model_name='roberta', model_type='roberta-base', top_k = 100, num_train_epochs = 1, BATCH_SIZE=32, learning_rate=2e-5, epsilon=1e-8, gradient_accumulation_steps = 1, max_grad_norm = 1, weight_decay = 0, number_of_warmup_steps=0, global_step = 0, tr_loss=0.0, logging_loss = 0.0  ):
    config_class, model_classifier, model_tokenizer = MODEL_CLASSES[model_name]
    tokenizer = model_tokenizer.from_pretrained(model_type)
    model = model_classifier.from_pretrained(model_type)
    model.cuda()
    model.train()
    save_model = model_name+'_t_'+str(top_k)+'_epoc_'+ str(num_train_epochs)+'_lr_'+ str(learning_rate)+'_b_s_'+ str(BATCH_SIZE)
    train_dataset, rerank = make_dataset_train(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, top_k, model_with_no_token_types, model_name=model_name)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    t_total = len(train_dataloader) // gradient_accumulation_steps * 3
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=number_of_warmup_steps, num_training_steps=t_total)
    model.zero_grad()
    for i in tqdm(range(num_train_epochs)):
      epoch_iterator = tqdm(train_dataloader, desc="Iteration")
      for step, batch in enumerate(epoch_iterator):
          model.train()
          batch = tuple(t.to(device) for t in batch)
          if model_name in model_with_no_token_types:
            inputs = {'input_ids':      batch[0],
                      #'token_type_ids': batch[1], 
                      'attention_mask': batch[1],
                      'labels':         batch[2]}
          else:
            inputs = {'input_ids':      batch[0],
                      'token_type_ids': batch[1], 
                      'attention_mask': batch[2],
                      'labels':         batch[3]}
          ouputs = model(**inputs)
          loss = ouputs[0]  
          loss.backward() 
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          tr_loss += loss.item()
          if (step + 1) % gradient_accumulation_steps == 0:
              optimizer.step()
              scheduler.step()  # Update learning rate schedule
              model.zero_grad()
              global_step += 1
    ## Dont Forget to save the model
    #torch.save(model.state_dict(), './saved_models/'+save_model+'state_dict'+'.pt')
    torch.save(model, './saved_models/'+save_model+'.pt')
    print("Model is saved as : ",save_model)
    print("Use this to load the model")
    return save_model
    
def evaluate_model(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, save_model, model_with_no_token_types, model_name='roberta',model_type='roberta-base', mode='train', top_k = 100 ):
    if mode=='train':
        train_dataset, rerank = make_dataset_train(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, top_k, model_with_no_token_types, model_name= model_name)
        train_dataloader = DataLoader(train_dataset, batch_size=top_k)
        model=torch.load('./saved_models/'+save_model+'.pt')
        preds = []
        with torch.no_grad(): 
          direct_aps = []
          reranked_aps = []
          epoch_iterator = tqdm(train_dataloader, desc="Iteration")
          for step, batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            if model_name in model_with_no_token_types:
              inputs = {'input_ids':      batch[0],
                        #'token_type_ids': batch[1], 
                        'attention_mask': batch[1]}
            else:
              inputs = {'input_ids':      batch[0],
                        'token_type_ids': batch[1], 
                        'attention_mask': batch[2]}
            outputs = model(**inputs)
            pred = outputs[0] 
            pred = np.argmax(nn.Softmax(dim=1)(pred).cpu().detach().numpy(), axis=1)
            pred = [rerank[step]['pred'][p] for p in range(len(rerank[step]['pred'])) if pred[p] != 0]
            direct_score = average_precision_score(rerank[step]['gold'], rerank[step]['pred'])
            reranked_score = average_precision_score(rerank[step]['gold'], pred)
            direct_aps.append(direct_score)
            reranked_aps.append(reranked_score)
        print(np.mean(np.array(direct_aps)))
        print(np.mean(np.array(reranked_aps)))
        if not os.path.exists('results/results_train.csv'):
            result_df = po.DataFrame(columns=['Model_train','TFIDF_MAP_train','Reranked_MAP_train'])
        else:
            result_df = po.read_csv('results/results_train.csv')
        results={'Model_train':save_model,'TFIDF_MAP_train':np.mean(np.array(direct_aps)),'Reranked_MAP_train':np.mean(np.array(reranked_aps))}
        result_df = result_df.append(results, ignore_index=True)
        result_df.to_csv('results/results_train.csv',index=False)
        
    elif mode=='dev':
        val_dataset, rerank = make_dataset_test(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, top_k, model_with_no_token_types, model_name= model_name)
        val_dataloader = DataLoader(val_dataset, batch_size=top_k)
        model=torch.load('./saved_models/'+save_model+'.pt')
        preds = []
        with torch.no_grad(): 
          direct_aps = []
          reranked_aps = []
          epoch_iterator = tqdm(val_dataloader, desc="Iteration")
          for step, batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            if model_name in model_with_no_token_types:
              inputs = {'input_ids':      batch[0],
                        #'token_type_ids': batch[1], 
                        'attention_mask': batch[1]}
            else:
              inputs = {'input_ids':      batch[0],
                        'token_type_ids': batch[1], 
                        'attention_mask': batch[2]}
            
            outputs = model(**inputs)
            pred = outputs[0]
            pred = np.argmax(nn.Softmax(dim=1)(pred).cpu().detach().numpy(), axis=1)
            pred = [rerank[step]['pred'][p] for p in range(len(rerank[step]['pred'])) if pred[p] != 0]
            preds.append(pred)
            direct_score = average_precision_score(rerank[step]['gold'], rerank[step]['pred'])
            reranked_score = average_precision_score(rerank[step]['gold'], pred)
            direct_aps.append(direct_score)
            reranked_aps.append(reranked_score)
        print(np.mean(np.array(direct_aps)))
        print(np.mean(np.array(reranked_aps)))
        if not os.path.exists('results/results_dev.csv'):
            result_df = po.DataFrame(columns=['Model_dev','TFIDF_MAP_dev','Reranked_MAP_dev'])
        else:
            result_df = po.read_csv('results/results_dev.csv')
        results={'Model_dev':save_model,'TFIDF_MAP_dev':np.mean(np.array(direct_aps)),'Reranked_MAP_dev':np.mean(np.array(reranked_aps))}
        result_df = result_df.append(results, ignore_index=True)
        result_df.to_csv('results/results_dev.csv',index=False)
        
def predict_model(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, save_model, model_with_no_token_types, top_k=100, model_name='roberta',model_type='roberta-base'):
    test_dataset, rerank = make_dataset_test(tokenizer,df, df_exp, uids, uid2idx, uid2text, ranks, preds, top_k, model_with_no_token_types, model_name= model_name)
    test_dataloader = DataLoader(test_dataset, batch_size=top_k) 
    model = torch.load('./saved_models/'+save_model+'.pt')
    with torch.no_grad(): 
      preds = []
      epoch_iterator = tqdm(test_dataloader, desc="Iteration")
      for step, batch in enumerate(epoch_iterator):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        if model_name in model_with_no_token_types:
          inputs = {'input_ids':      batch[0],
                    #'token_type_ids': batch[1], 
                    'attention_mask': batch[1]}
        else:
          inputs = {'input_ids':      batch[0],
                    'token_type_ids': batch[1], 
                    'attention_mask': batch[2]}
        outputs = model(**inputs)
        pred = outputs[0]
        pred = np.argmax(nn.Softmax(dim=1)(pred).cpu().detach().numpy(), axis=1)
        pred = [rerank[step]['pred'][p] for p in range(len(rerank[step]['pred'])) if pred[p] != 0]
        preds.append(pred)
    uids = df_exp.uid.apply(remove_combo_suffix).values
    qids = df.QuestionID.tolist()
    preds_idx = []
    for i in tqdm(range(len(preds))):
      question_id = qids[i]
      for p in preds[i]:
        explanation_uid = df_exp.loc[df_exp['text'] == p, 'uid'].to_list()[0]
        preds_idx.append(question_id + "\t" + explanation_uid)
    print("The predictions are stored in the file : "+'./predictions/'+save_model+'.txt')
    write_preds(preds_idx, './predictions/'+save_model+'.txt')
    
