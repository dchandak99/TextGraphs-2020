# ChiSquareX at TextGraphs-14: COLING 2020

## Multi-Hop Inference for Explanation Regeneration
This is our attempt of the shared task on **Multi-Hop Inference for Explanation Regeneration** at the [TextGraphs-14 workshop](https://sites.google.com/view/textgraphs2020), part of Coling 2020.

The workshops in the TextGraphs series have published and promoted the synergy between the field of Graph Theory and Natural Language Processing. Besides traditional NLP applications like word sense disambiguation and semantic role labeling, and information extraction graph-based solutions nowadays also target new web-scale applications like information propagation in social networks, rumor proliferation, e-reputation, language dynamics learning, and future events prediction, to name a few.  Previous editions of the series can be found [here](http://textgraphs.org/).

Official repository of the task can be found [here](https://github.com/cognitiveailab/tg2020task).

## Introduction  
Multi-hop inference is the task of combining more than one piece of information to solve an inference task, such as question answering. This can take many forms, from combining free-text sentences read from books or the web, to combining linked facts from a structured knowledge base.

The Explanation Regeneration shared task asked participants to develop methods to reconstruct gold explanations for elementary science questions (Clark et al., 2018), using a new corpus of gold explanations (Jansen et al., 2018) that provides supervision and instrumentation for this multi-hop inference task. Each explanation is represented as an “explanation graph”, a set of atomic facts (between 1 and 16 per explanation, drawn from a knowledge base of 5,000 facts) that, together, form a detailed explanation for the reasoning required to answer and explain the resoning behind a question. Linking these facts to achieve strong performance at rebuilding the gold explanation graphs requires methods to perform multi-hop inference - which has been shown to be far harder than inference of smaller numbers of hops (Jansen, 2018), particularly for the case here, where there is considerable uncertainty (at a lexical level) of how individual explanations logically link somewhat ‘fuzzy’ graph nodes. The explanations include both core scientific facts as well as detailed world knowledge, allowing this task to appeal to those interested in both multi-hop reasoning and common-sense inference.  

## Result  
Results of different models on the test, train and dev dataset can be found here:
The results have been in terms of the **MAP** scores.  

|  Method          | Train     | Dev    | Test     | 
|------------------|-----------|--------|----------|
|    tf hub        |     0.321 |   0.34 |     0.31 |
|Sentence BERT     |     0.24  |  0.253 |          |
|tf-idf (sublinear)|           |  0.343 |     0.33 |
| Recursive tf.idf |           |  0.477 |          |
|    BERT          |           |        |          |
|    BART          |           |        |          |
|   Longformer     |           |        |          |
|    Scibert       |           |        |          |
|    Roberta       |           |        |          |

Our Final Leaderboard Test MAP: **0.4902**

## Reproducing Results  
### File descriptions:  
Supplementary files including data and trained models can be found here (Link to be added).  
[baseline_model.py](baseline_model.py):  Uses a baseline that uses a term frequency model (tf.idf) to rank how likely table row sentences are to be a part of a given explanation. The performance of this baseline on the development partition is 0.255 MAP.  
[utils.py](utils.py):  
[paths.py](paths.py):  
[rank.py](rank.py):  
[train_and_eval_functions.py](train_and_eval_functions.py):  
[RUN_model.ipynb](RUN_model.ipynb):  

### How to Run:  
Instructions on how to run, and meaning of the functions have been given in this interative [notebook](RUN_model.ipynb).

## Acknowledgements  
A huge thanks to the organizers Peter Jansen and Dmitry Ustalov for curating this super interesting task.  

A big thanks to **Hugging Face** for their excellent [transformers](https://github.com/huggingface/transformers) repository. A lot of the code is built on top of their examples.
