# A 2020 Guide to Named Entity Recognition

In this article we will give you a brief overview of Named Entity Recognition (NER), its importance in information extraction, its brief history, latest approaches used to perform NER and at the end will also show you how to quickly use a latest NER model, on your dataset.

## What is NER?

As per Wikipedia, a named entity is a real world object (can be abstract or have a physical existence) such as persons, locations, organizations, products, etc. that can be denoted with a proper name e.g. in *Ashish Vaswani et.al introduced Transformers architecture in 2017.* *Ashish Vaswani* and *2017* could be considered named entities since *Ashish Vaswani* (phsically exist) is name of a person & *2017* (abstract) is a date. And NER is a part of information extraction task where the goal is to locate and classify such named entities from unstrctured text into predefined entity categories e.g. in *Allen Turing was born on 23 June 1912.* *Allen Turing* and *23 June 1912* could be considered a *person entity* belonging to person category and and *23 June 1912* could be considered a *date entity* belonging to date category.

Formally, given a sequence of tokens $s={w_1,w_2,...,w_n}$, an NER task is to output a list of tuples ${I_s,I_e,t}$, each of which is a named entity mentioned in $s$. Here, $I_s ∈ [1, N], I_e ∈ [1, N]$ are the start and the end indexes of a named entity mention; $t$ is the entity type from a predefined category set. 

!["NER DEFINITION"](./ner_definition.jpg)
**[source](https://sauravvmaheshkar.gitbook.io/saurav-maheshkar/projects/named-entity-recognition-using-reformers/named-entity-recognition-a-brief-overview)**

In the above definition, we are assigning a single entity category to an entiy, but in [recent literature](https://www.aclweb.org/anthology/E17-1075.pdf) classifying entities into multiple categories using datesets like [Wiki (Gold)](https://github.com/juand-r/entity-recognition-datasets), [BBN](https://catalog.ldc.upenn.edu/LDC2005T33) etc. is also quite popular and classified as fine-grained NER task. Another, popular variant of NER task is nested NER tasks, where some entity mentions are embedded in longer entity mentions. e.g. *Bank of Americs* where *Bank of America* and *America* both could be enties. Extracting such entities could be a bit tricky and we will look at few architectures to handle that but let us first see where NER is useful. 


## Why or Where NER is useful?

Extracting named entities from unstructured text could be very useful step for a variety of downstream tasks and in supporting various applications. We will look at a few of the tasks and applications below.
- In chatbots extracting entities is very essential to understand the context and provide relevant recommendation/information based on mentioned entity. In [Rasa](https://rasa.com/), a popular chatbot development framework, there is a separate NLU (Natural Language Understanding) pipeline for training an entity detection and intent classification model.
- Semantic text search can be improved by recognizing named entities in search queries which in turn enable search engines to understand the concepts, meaning and intent behind the quries well.
- Categorizing news articles into pre-defined hierarchies using different entities related to sports, politics, etc.
- Extracted entities could used as features in other downstream tasks like question answering, machine translation etc.

This is not the exhaustive list of use cases of NER but you can understand that usefulness of NER is versatile.

## Brief History of NER

Before jumping to the latest deep learning based techniques to perform NER it is important to get a brief overview of traditional approaches to NER. So let us see how it started.

The term “Named Entity” (NE) was first used at the sixth Message Understanding Conference (MUC-6), as the task of identifying names of organizations, people and geographic locations in text, as well as currency, time and percentage expressions in 1996. Since then there has been a lot of work and NER has become a very popular and important NLP task.

### Rule Based Methods
In the beginning there were rule based methods which relied on hand crafted rules. Rules are designed on domain-specific gazetteers and syntactic-lexical patterns. They work very well when lexicon is exhaustive.

### Unsupervised Approaches
Some unsupervised approaches like clustering-based NER systems which extract named entities from the clustered groups based on context similarity, systems which uses terminologies, corpus statistics (e.g. inverse document frequency) and shallow syntactic knowledge (e.g. noun phrase chunking) etc. have been made.

### Supervised Approaches
While applying supervised machine learning approaches, NER problem is cast to a multi-class classification or sequence labelling task. In this approach, features are carefully designed using word-level features, document, corpus features etc. to represent each training example. Machine learning algorithms are then utilized to learn a model to recognize similar patterns from unseen data.


### Deep Learning Approaches

Deep Learning has revolutionised every field it has been applied into, and same is the case in NER. Current state of the art models in NER are all powered by deep learning architectures. There are lots of reason behind the success of deep learning models and we will look at a few of both general and specific reasons here. 
- In classical machine learning problems features are created manually or using domain expertise. And a machine learning algorithm like SVM, Random Forest etc are applied on it. This could introduce bias specific to a dataset, and also sometimes creating good features (that can be classified easily) is not possible e.g. features of different objects in an image. On the other hand deep learning models creates necessary features on its own using only the raw data. The features or repsentations created by deep learning models capture a lot of complex semantic features which would be difficult for even an expert to design using logical rules. And this helps them in getting overall performance.
- Deep Learning models are data hungry. They are trained on lots of examples before they can learn extract relevant features. Because of abundance of data from internet and sufficient computing power to process it, these models have been setting new SOTA benchmarks. Specifically, even if the datasets are small in NER, deep learning models are built on top of big language models which have been trained on large amount of data, which is the main reason behind the performance boost.
- Another really important factor/choice which changed the NLP demography completely is distributed representation of words or characters. In order to perform machine learning on textual data it needs to be converted into some numerical representation, and previously one hot enoded vectors were used to represent words i.e. a word 'deep' will be represented by a vector $[0,0,0,...,1,0,...,0]$ where the size of the vector is equal to vocab size. While distributed word representations like word2vec, glove etc. are low dimensinal in comparison to one hot encoded representations and captures semantic and syntactic properties of words. This succesively helps deep lernig models as well. Another popular and recent variant of this is contextual distributed representions. In contextual distributed representions, context in which a word appears is also taken into consideration e.g. in *I like to eat apple.* and *I bought apple stocks* word *apple* has different meaning so its vector representation should also be different depending on the context it appears in.
- Attention is one such idea in deep learning which makes almost anything work :P. It has made difficult problems like machine translation, question answering etc. work really well in real world. Though there are not popular NER model which uses attention, one popular neural network architecture, [Transformers](https://arxiv.org/abs/1706.03762), which we will be discussing below, is based on a variant of attention called self-attention, and it has changed the way we do NLP a lot. If you are not familiar with attention and self-attention go through this [article](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) by Denny Britz, and this [article](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar before proceeding. In brief, attention is a mechanism to dynamically assign coefficients (or weightage) in a linear combination of $n$ different things (scalars, vectors, matrices etc.). Intuitively it means while combining $n$ things, attention helps us decide their individual contributions (or importance). Here dynamically means that the coefficients will change according to each fed example. Self-attention also serves the same purpose as attention with only differnce being the way it is calculated. Please refer to [this](https://www.reddit.com/r/LanguageTechnology/comments/be6jfc/what_is_the_difference_between_self_attention_and/el3qr16/?context=8&depth=9) reddit thread to understand the difference better.

After going through few of the most important ideas in deep learning, let us go through a few popular and a few latest deep learning architectures used for NER.

### 

### Bi-LSTM with CRF

!["Bi-LSTM with CRF"](./bilstm_crf.jpg)
**Bi-LSTM with CRF. [source](https://arxiv.org/pdf/1603.01360.pdf)**

Sequence tagging using Bi-LSTM (or LSTM) has been explored before where combination of forward and backward embeddings of each token is passed to a linear classifier (sometimes an additional linear transformation operation is added before linear classifier) which produces a probability distribution over all the possible entity tags for each token. However, this approach doesn't work well when there are strong dependencies across output labels. NER is one such task, since the “grammar” that characterizes interpretable sequences of tags imposes several hard constraints e.g., I-PER cannot follow B-LOC. Therefore, instead of modeling tagging decisions independently, modelling them jointly using a conditional random field works very well. The CRF layer could add some constrains to the final predicted labels to ensure they are valid. These constrains can be learned by the CRF layer automatically from the training dataset during the training process. Now let us try to understand the formulation.

First a score $s$ is calculated for a possible sequences of tags $\bold{y}$ and a given input sentence $\bold{X}$ using below equation:

$s(\bold{X},\bold{y}) = \sum_{i=0}^{n} A_{y_i,y_{i+1}} + \sum_{i=1}^{n}P_{i,y_{i}}$

where $n$ is the number of words/tokens in a sentence, $\bold{A}∈R^{(k+2) \times (k+2)}$ is a matrix of transition scores such that $A_{i,j}$ represents the score of a transition from the entity tag $i$ to entity tag $j$, $(k+2)$ is the number of entity tags + start and end tags of a sentence, $\bold{P}∈R^{n \times k}$ the matrix of scores output by the bidirectional LSTM network and $P_{i,j}$ corresponds to the score of $j^{th}$ tag of the $i^{th}$ word in a sentence, $\bold{X}=(\bold{x_1},\bold{x_2},...,\bold{x_n})$ is an input sentence, and $\bold{y}=(y_1,y_2,...,y_n)$ are entity tags for a sequence of predictions.

A softmax over all possible tag sequences yields a probability for the sequence $\bold{y}$:

$p(\bold{y}|\bold{X})=\frac{e^{s(\bold{X},\bold{y})}}{\sum_{\hat{y}∈\bold{Y_x}}e^{s(\bold{X},\bold{\hat{y}})}}$

And during training log-probability of the correct tag sequence is maximized:

$log(p(\bold{y}|\bold{X}))= s(\bold{X},\bold{y}) - log(\sum_{\hat{y}∈\bold{Y_x}}e^{s(\bold{X},\bold{\hat{y}})})$

where $\bold{Y_x}$ represents all possible tag sequences for a sentence $\bold{X}$.

This formulation encourages the network to produce a valid sequence of output labels.

During evaluation the sequence that obtains the maximum score is predicted:

$\bold{y}^* = \argmax_{\hat{y}∈\bold{Y_x}} s(\bold{X},\bold{\hat{y}})$


### NER using BERT like Models

Doing NER with different BERT like models like RoBERTa, Distill-BERT, AlLBERT etc is very simillar. We will see this approach here briefly since already it has been covered in detail [before](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/). 

Basically, a linear classifier is added on top of [BERT](https://arxiv.org/abs/1810.04805) or BERT like models to classify each token into an entity category. 

!["BERT NER"](./bert_ner.jpg)
**BERT NER. [source](https://arxiv.org/abs/1810.04805)**

Embedding matrix, $E∈R^{n \times d}$ where $d$ is the vector dimension of the last layer of the model and $n$ is the number of tokens obtained from BERT model is fed to a linear classifier which return a probability distribution over all the entity categories. And the complete architecture with classifier head is finetuned on cross-entropy loss. During validation entity category with maximum probability is assigned to each token.

Now let us go through a few latest techniques to do NER.

### Named Entity Recognition as Machine Reading Comprehension

In recent years( e.g. https://arxiv.org/pdf/2011.03023.pdf), we have seen formalizing NLP problems as question answering tasks has shown very good performance using relatively less data. In [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/pdf/1910.11476v6.pdf) the authors have tried to implement NER as a MRC problem and had been able to achieve very good results, even on nested NER datasets using very litle finetuning of BERT language model. So, let us dig into the model architecture and try to understand the training procedure. 

For question answering problem we feed the question *q* and the passage *X* concatenated, forming the combined string *{[CLS], q1, q2, ..., qm, [SEP], x1, x2, ..., xn}*, where *q1,q2,..,qm* and *x1,x2,..,xm* are tokenized question and passage respectively and *[CLS]* & *[SEP]* are special tokens, to BERT, which returns us a context representation matrix $E∈R^{n \times d}$ where *d* is the vector dimension of the last layer of BERT and the query representation is simply dropped. Using the matrix $E$ probabibility of each token being start and end indices are calculated respectively applying below operations: 

$P_{start} = softmax_{each row}(E.T_{start})∈R^{n \times 2}$

and 

$P_{end} = softmax_{each row}(E.T_{end})∈R^{n \times 2}$

where $T_{start}$ and $T_{end} ∈ R^{n \times 2}$ are the weights learned during training.  

**NOTE: Though $T_{end}$ or $T_{start} ∈ R^{n \times 2}$ have been mentioned in [paper](https://arxiv.org/pdf/1910.11476v6.pdf), in the implementation $T_{end}$ or $T_{start} ∈ R^{n \times 1}$ with sigmoid have been used which is certainly more efficient in terms of compute.**

After above operation applying argmax to each row of $P_start$ and $P_end$ gives us the predicted indexes that might be the starting or ending positions, i.e. $\hat{I}_{start}$ and $\hat{I}_{end}$:

$\hat{I}_{start}$ = $\{i|argmax(P^{i}_{start})=1, i=1,..,n \}$

$\hat{I}_{end}$ = $\{i|argmax(P^{i}_{end})=1, i=1,..,n \}$

where the superscript $i$ denotes the i-th row of a matrix.

In the next step each predicted start index needs to be matched with its corresponding predicted end index. A binary classification model is trained to predict the probability of a pair of start and end indices matching as follows:

$P_{i_{start},j_{end}} = sigmoid(m.concat(E_{i_{start}},E_{j_{end}}))$

where $i_{start}∈$ $\hat{I}_{start}$, $i_{end}∈$ $\hat{I}{end}$, and $m∈R^{1 \times 2}$ are the weights learned during training.

The sum of cross-entropy loss (or binary cross-entropy loss) on $P_{start}$ & $P_{end}$ and binary cross-entropy loss on $P_{i_{start},j_{end}}$ are calculated and gradients are backpropagated to train the complete architecture. In this method, more than one entity and even nested entities are extracted easily in comparison to [traditional question answering architecture](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/#part-1-how-bert-is-applied-to-question-answering) where only a single span is output given a query and passage pair. Another benefit of this approach is sample efficient training i.e. less training data is required to train a decent performing model. 

### LUKE

Using distributed entity representations just like word embeddings is another method which has been explored before to improve NLP tasks involving entities e.g. relation classification, entity typing, NER. LUKE (Language Understanding with Knowledge-based Embeddings) is an extension of this method by tweaking BERT's Masked Language Model (MLM) and adding a new entity-aware self-attention mechanism. Let us start with the architecture. 

!["LUKE Architecture"](./luke_architecture.jpg)

In [LUKE](https://arxiv.org/pdf/2010.01057v1.pdf) the aim is to create contextual representations of entities just like contextual word representations. Hence, entities appended at the end of word tokens are fed to [RoBERTa](https://arxiv.org/abs/1907.11692) as shown in the figure above. In order to accomodate entities new embeddings, token embeddings for entities, position embeddings for entities and entity type embeddings are introduced. 
- **Token Embeddings for Entities**: Just like word token embeddings, entity token embeddings create a dense representation of each entity. To handle large number of entities, entity token embedding matrix is decomposed into two small matrices $B∈R^{V_e \times H}$ and $U∈R^{H \times D}$ where $V_e$ is number of tokens, $H$ intermediate dimension and $D$ is the token embedding dimension.
- **Position Embedding for Entities**: A separate position embedding matrix $D_i∈R^D$, where $D_i$ is a position embedding of an entity appearing at $i$-th position in the sequence. If an entity name contains multiple words, its position embedding is computed by averaging the embeddings of the corresponding positions, as shown above.
- **Entity Type Embedding**: This embedding represents that the token is an entity. It is a single vector denoted by $e∈R^D$.
This setting is trained on Masked Language Modelling objective with a slight variation in masking entities i.e. complete entity e.g. Los Angeles is masked instead of just token masking in standard MLM objective, and loss calculation i.e. sum of standard MLM loss and cross-entropy loss on predicting the masked entities.

Another variation, in the Query matrix of self-attention layer i.e. different query matrices for different types of tokens, is tried out and showed strong gains in ablation studies.

!["LUKE Query Matrices"](./luke_query_matrix.jpg)

where $Q_{w2e},Q_{e2w},Q_{e2e}∈R^{L \times D}$ are query matrices and $L$ is intermediate query, key and value representation dimension.

LUKE has been tested on different problems and showed SOTA results on many of those tasks.

!["LUKE All Results"](./luke_all_results.jpg)
**Results of LUKE on ReCoRD, CoNLL-2003, (first row) SQuADv1, TACRED (second row) dataset. [source](https://arxiv.org/pdf/2010.01057v1.pdf)**

Though LUKE is showing very good results on many NLP tasks, comparison of its contextual entity embeddigs with previous static and contextual embeddings would be really interesting to see and may justify the additional space and time complexity.


## Datasets

There are many NER datasets including domain specific datasets freely available. I will briefly explain a few of the popular ones which are frequently used in current research papers.
- **CoNLL-2003**: It contains annotations for Reuters news in two languages: English and German. The English dataset has a large portion of sports news with annotations in four entity types (Person, Location, Organization, and Miscellaneous). The complete dataset can be downloaded from [here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003).
- **OntoNotes**:  The OntoNotes data is annotated on a large corpus, comprising of various genres (weblogs, news, talk shows, broadcast, usenet newsgroups, and conversational telephone speech) with structural information (syntax and predicate argument structure) and shallow semantics (wordsense linked to an ontology and coreference). There are 5 versions, from Release 1.0 to Release 5.0. The texts are annotated with 18 entity types, consisting of 11 types (Person, Organization, etc) and 7 values (Date, Percent, etc). This dataset can be downloaded from [here](https://catalog.ldc.upenn.edu/LDC2013T19).
- **ACE 2004 and ACE 2005**: The corpus consists of data of various types annotated for entities and relations and was created by Linguistic Data Consortium. The two datasets each contain 7 entity categories. For each entity type, there are annotations for both the
entity mentions and mention heads. These datasets contain about 24% and 22% of the nested mentions. And it can be downloaded from [here](http://cogcomp.github.io/cogcomp-nlp/corpusreaders/doc/ACEReader.html)

!["Sample ACE-2004 dataset"](./ace2004.jpg)
*Sample ACE-2003 dataset [source](http://cogcomp.github.io/cogcomp-nlp/corpusreaders/doc/ACEReader.html)*


## Metrics

Below are the most commonly used evaluation metrics for NER systems. There are other complex evaluations metrics also available but they are not intuitive and make error analysis difficult.
- **Precision**: Precision is defined as below 
$Precison=\frac{TP}{TP+FP}$, where
$TP$ = True Positive, i.e.  entities that are recognized by NER and match the ground truth.
$FP$ = False Positive, i.e. entities that are recognized by NER but do not match the ground truth.
Precision measures the ability of a NER system to present only correct entities.
- **Recall**: Recall is defined as below
$Recall=\frac{TP}{TP+FN}$, where
$FN$ = False Negative, i.e. entities annotated in the ground which that are not recognized by NER.
Recall measures the ability of a NER system to recognize all entities in a corpus.
- **F-1 Score**: F-1 score is the harmonic mean of precision and recall. i.e.
$F-1 Score = 2*\frac{Precision*Recall}{Precision+Recall}$
Since most NER systems involve multiple entity types, it is often required to assess the performance across all entity classes. Two measures
are commonly used for this purpose: the macroaveraged F-1 score and the micro-averaged F-1 score. The macro-averaged F-1 score computes the F-1 score independently for each entity type, then takes the average (hence treating all entity types equally). The micro-averaged F-1 score aggregates the contributions of entities from all classes to compute the average (treating all entities equally).


## Train your own NER model

Now let us train our own NER model. We will be using ACE 2004 dataset to train an MRC based NER model, as discussed above. So let us look at different steps involved. We will be using this repository as a reference for our implementation and complete code can be found on repo [here](https://github.com/SKRohit/mrc-for-flat-nested-ner).

### Data Preparation

ACE 2004 dataset can be downloaded from [here](https://cogcomp.seas.upenn.edu/page/resource_view/60). It is structured in xml format like below and we need to process the data in order to convert it into proper structure to feed to underlying BERT model as a question answering problem. Fortunately the processed version of data which looks like below can be downloaded from [here](https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md). But if you want to use your own data make sure to convert it into relevant format. A sample code to do so is provided [here](https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/genia2mrc.py).

!["Processed ACE 2004 DATA"](./processesed_ace2004data.jpg)

### Training

We have already seen the architecture of the model and how it is trained above. So, here we will look at the code and different hyperparameters to start the training process. First clone this [repository](https://github.com/SKRohit/mrc-for-flat-nested-ner) 
```
git clone https://github.com/SKRohit/mrc-for-flat-nested-ner
cd mrc-for-flat-nested-ner
```
Then run the below command to start training the Huggingface's 'bert-base-uncased' implementation on ACE-2004 data present in `data` directory. One can play around with different parameters as well.

```
python trainer_new.py --data_dir "./data/ace2004" \
--bert_config_dir "./data" \
--max_length 128 \
--batch_size 4 \
--gpus="1" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr 3e-5 \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 2 \
--default_root_dir "./outputs" \
--mrc_dropout 0.1 \
--bert_dropout 0.1 \
--max_epochs 20 \
--span_loss_candidates "pred_and_gold" \
--weight_span 0.1 \
--warmup_steps 0 \
--max_length 128 \
--gradient_clip_val 1.0 \
--model_name 'bert-base-uncased'
```

In the original repository, pre-trained BERT models by Google were used, which needed to be downloaded and converted into PyTorch models before being used. Hence, here I adapted the code to run any Huggingface's BERT implementation by just providing the model name in `model_name` parameter in `trainer.py`.