# DeepHateExplainer
Source codes and supplementary for the paper "DeepHateExplainer: Explainable Hate Speech Detection in Under-resourced Bengali Language", under review in IEEE Access journal.The preprint version can be found and read on arXiv (https://arxiv.org/ftp/arxiv/papers/2012/2012.14353.pdf) as well. 

#### Methods
In our approach, Bengali texts are first comprehensively preprocessed, before classifying them into political, personal, geopolitical, and religious hates, by employing the neural ensemble method of different transformer-based neural architectures (i.e., monolingual Bangla BERT-base, multilingual BERT-cased/uncased, and XLM-RoBERTa). Subsequently, important (most and least) terms are identified with sensitivity analysis and layer-wise relevance propagation (LRP), before providing human-interpretable explanations. Finally, to measure the quality of the explanation (i.e., faithfulness), we compute the comprehensiveness and sufficiency. 

#### Results
Evaluations against machine learning~(linear and tree-based models) and deep neural networks (i.e., CNN, Bi-LSTM, and Conv-LSTM with word embeddings) baselines yield F1 scores of 84%, 90%, 88%, and 88%, for political, personal, geopolitical, and religious hates, respectively, outperforming both ML and DNN baselines. 

#### Data collections
We extend the Bengali Hate Speech Dataset ((https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset)) with additional 3,000 labelled samples. The Bengali Hate Speech Dataset categorized into political, personal, geopolitical, religious, and gender abusive hates. However, our empirical study and linguist analysis observe that distinguishing personal from gender abusive hate is often not straightforward, as they often semantically overlap. To justify this, our study observes that distinguishing personal hates from gender abusive hates was very challenging. Often statements that express hatred statement towards a person commonly used Benglai words that are directed mostly towards women. Therefore, ww follow the bootstrap approach for the data collection, where specific types of texts containing common slurs and terms, either directed towards a specific person or entity or generalized towards a group, are only considered. Texts were collected from Facebook, YouTube comments, and newspapers. We categorize the samples into political, personal, geopolitical, and religious hate. 

#### Data availability
We made publicly available this dataset (https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset), but only for research purposes. So, if you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{karim2020BengaliNLP,
        title={Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
        author={Md. Rezaul Karim, Bharathi Raja Chakravarti, John P. McCrae, and Michael Cochez},
        conference={7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020)},
        year={2020}
    }

#### Availability of pretrained models
We plan to make public all the pretrained models and some computational resources available, but it will take time. 

### Instruction to run the codes and sample results:
    -- ```git clone https://github.com/rezacsedu/DeepHateExplainer.git```
    -- 
    -- ```cd xlm-roberta```
    -- ```python3 run.sh``` file 

Below we outlined some sample results. Your results may vary given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

#### 5-fold bangla-bert-base[pooled output] (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
```
Accuracy:: 0.8956180976131101
Precision:: 0.8966952209031919
Recall:: 0.8956180976131101
F_score:: 0.8958846395067238
classification_report::                
                  precision    recall  f1-score   support
           0       0.86      0.91      0.88       793
           1       0.88      0.87      0.88       549
           2       0.87      0.86      0.87       465
           3       0.95      0.92      0.93      1000

    accuracy                           0.90      2807
   macro avg       0.89      0.89      0.89      2807
weighted avg       0.90      0.90      0.90      2807
[Last two hidden layer] max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5
Accuracy:: 0.8941930887068044
Precision:: 0.8956740117332456
Recall:: 0.8941930887068044
F_score:: 0.8944349193898055
classification_report::                
               precision    recall  f1-score   support
           0       0.85      0.91      0.88       793
           1       0.89      0.85      0.87       549
           2       0.87      0.86      0.87       465
           3       0.95      0.92      0.93      1000

    accuracy                           0.89      2807
   macro avg       0.89      0.89      0.89      2807
weighted avg       0.90      0.89      0.89      2807

[Last four hidden layer] max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5
Accuracy:: 0.8924118275739223
Precision:: 0.8928764160112657
Recall:: 0.8924118275739223
F_score:: 0.8924472225081573
classification_report::                precision    recall  f1-score   support

           0       0.87      0.91      0.89       793
           1       0.88      0.88      0.88       549
           2       0.86      0.83      0.85       465
           3       0.94      0.92      0.93      1000

    accuracy                           0.89      2807
   macro avg       0.89      0.88      0.88      2807
weighted avg       0.89      0.89      0.89      2807

[Custom layer] max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5
Accuracy:: 0.8916993231207695
Precision:: 0.8920307371175132
Recall:: 0.8916993231207695
F_score:: 0.8916912321325475
classification_report::                precision    recall  f1-score   support

           0       0.86      0.89      0.88       793
           1       0.90      0.86      0.88       549
           2       0.87      0.85      0.86       465
           3       0.93      0.93      0.93      1000

    accuracy                           0.89      2807
   macro avg       0.89      0.88      0.89      2807
weighted avg       0.89      0.89      0.89      2807
```
#### 5-fold m-bert-base-cased[pooled output] (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
```
Accuracy:: 0.8685429283933025
Precision:: 0.8698164487534069
Recall:: 0.8685429283933025
F_score:: 0.868620288657791
classification_report::                
                precision    recall  f1-score   support
           0       0.83      0.88      0.86       793
           1       0.87      0.82      0.85       549
           2       0.85      0.90      0.87       465
           3       0.91      0.87      0.89      1000

    accuracy                           0.87      2807
   macro avg       0.87      0.87      0.87      2807
weighted avg       0.87      0.87      0.87      2807

[Last two hidden layer] max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5
Accuracy:: 0.8685429283933025
Precision:: 0.8698164487534069
Recall:: 0.8685429283933025
F_score:: 0.868620288657791
classification_report::               
                precision    recall  f1-score   support
           0       0.83      0.88      0.86       793
           1       0.87      0.82      0.85       549
           2       0.85      0.90      0.87       465
           3       0.91      0.87      0.89      1000

    accuracy                           0.87      2807
   macro avg       0.87      0.87      0.87      2807
weighted avg       0.87      0.87      0.87      2807
```
#### 5-fold m-bert-base-uncased[pooled output] (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
```
Accuracy:: 0.8963306020662629
Precision:: 0.8974252974611797
Recall:: 0.8963306020662629
F_score:: 0.896590187463368
classification_report::                precision    recall  f1-score   support

           0       0.86      0.91      0.88       793
           1       0.88      0.85      0.87       549
           2       0.87      0.88      0.88       465
           3       0.95      0.92      0.93      1000

    accuracy                           0.90      2807
   macro avg       0.89      0.89      0.89      2807
weighted avg       0.90      0.90      0.90      2807
```
#### 5-fold xlm_roberta_large [pooled output] (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
```
Accuracy:: 0.8984681154257214
Precision:: 0.8992484246664187
Recall:: 0.8984681154257214
F_score:: 0.8987235449924004
classification_report::               
              precision    recall  f1-score   support
           0       0.88      0.88      0.88       793
           1       0.90      0.89      0.89       549
           2       0.84      0.88      0.86       465
           3       0.94      0.93      0.93      1000

    accuracy                           0.90      2807
   macro avg       0.89      0.89      0.89      2807
weighted avg       0.90      0.90      0.90      2807
```
### Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{karim2020BengaliNLP,
        title={Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
        author={Md. Rezaul Karim, Bharathi Raja Chakravarti, John P. McCrae, and Michael Cochez},
        conference={7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020)},
        year={2020}
    }
    
      @article{karim2020deephateexplainer,
      title={DeepHateExplainer: Explainable Hate Speech Detection in Under-resourced Bengali Language},
      author={Karim, Md and Dey, Sumon Kanti and Chakravarthi, Bharathi Raja and others},
      journal={arXiv preprint arXiv:2012.14353},
      year={2020}
    }

### Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de

