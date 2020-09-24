# TextGraphs-14: COLING 2020

## Multi-Hop Inference for Explanation Regeneration
This is our attempt of the shared task on **Multi-Hop Inference for Explanation Regeneration** at the [TextGraphs-14 workshop](https://sites.google.com/view/textgraphs2020), part of Coling 2020.

The workshops in the TextGraphs series have published and promoted the synergy between the field of Graph Theory and Natural Language Processing. Besides traditional NLP applications like word sense disambiguation and semantic role labeling, and information extraction graph-based solutions nowadays also target new web-scale applications like information propagation in social networks, rumor proliferation, e-reputation, language dynamics learning, and future events prediction, to name a few.  Previous editions of the series can be found [here](http://textgraphs.org/).

Official repository of the task can be found [here](https://github.com/cognitiveailab/tg2020task).

### Introduction:  
The Explanation Regeneration shared task asked participants to develop methods to reconstruct gold explanations for elementary science questions (Clark et al., 2018), using a new corpus of gold explanations (Jansen et al., 2018) that provides supervision and instrumentation for this multi-hop inference task. Each explanation is represented as an “explanation graph”, a set of atomic facts (between 1 and 16 per explanation, drawn from a knowledge base of 5,000 facts) that, together, form a detailed explanation for the reasoning required to answer and explain the resoning behind a question. Linking these facts to achieve strong performance at rebuilding the gold explanation graphs requires methods to perform multi-hop inference - which has been shown to be far harder than inference of smaller numbers of hops (Jansen, 2018), particularly for the case here, where there is considerable uncertainty (at a lexical level) of how individual explanations logically link somewhat ‘fuzzy’ graph nodes.

### Result:
Results of different models on the test, train and dev dataset can be found here:
The results have been in terms of the **MAP** scores.  

|  Method          | Train     | Dev    | Test     | 
|------------------|-----------|--------|----------|
|    tf hub        |     0.321 |   0.34 |     0.31 |
|Sentence BERT     |     0.24  |  0.253 |          |
|tf-idf (sublinear)|           |  0.343 |     0.33 |
| Recursive tf.idf |           |  0.477 |          |
|    tf hub        |     0.321 |   0.34 |     0.31 |
|    tf hub        |     0.321 |   0.34 |     0.31 |

### Last Year:  
  
A paper summarizing the task and the submissions can be found [here](https://www.aclweb.org/anthology/D19-5309.pdf)  
  
The 4 submissions last year:  
- Explanation ReGeneration using Language Models and Iterative Re-Ranking: [Paper](https://www.aclweb.org/anthology/D19-5310.pdf)  
- Red Dragon AI: Language Model Assisted Explanation Generation: [Paper](https://www.aclweb.org/anthology/D19-5311.pdf) and [Code](https://github.com/mdda/worldtree_corpus/tree/textgraphs)  
- Team SVMrank: Leveraging Feature-rich Support Vector Machines: [Paper](https://www.aclweb.org/anthology/D19-5312.pdf) and [Code](https://github.com/jenlindadsouza/tg2019task)   
- Chains-of-Reasoning: Reasoning over Chains of Facts for Explainable Multi-hop Inference: [Paper](https://www.aclweb.org/anthology/D19-5313.pdf) and [Code](https://github.com/ameyagodbole/multihop_inference_explanation_regeneration)   
  
### Links:  
More information about the task held in TextGraphs 2020 can be found here:  
  
- https://competitions.codalab.org/competitions/23615 (Overview and Submission)  
- https://competitions.codalab.org/forums/20311/ (Forums)  
- https://github.com/cognitiveailab/tg2020task (Instructions and Baseline)  
- This is the link to forum/discussion with officials. Any questions please refer this. https://competitions.codalab.org/forums/20311/3929/
