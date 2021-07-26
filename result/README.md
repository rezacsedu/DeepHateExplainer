### Evaluation results
Here, we enlist evaluation results of each BERT variant and ensemble of tope models on held-out test set. Please note the following class encoding to interprete the class-specific classification reports:
 
-	Personal hate: class 0
-	Political hate: class 1
-	Religius hate: class 2
-	Geopolitical hate: class 3.

#### Model: Bangla-BERT
Parameters: max_len=128, train_batch=16, test_batch=32, epochs = 6, lr=3e-5
```
Accuracy: 0.8625
Precision: 0.8632826723106086
Recall: 0.8625
F1-score: 0.8624422569771956
MCC: 0.7997196872300161
---------------------------------------------------------
Class-wise classification report:               
               precision    recall  f1-score   support

           0       0.90      0.89      0.89       524
           1       0.81      0.75      0.77       157
           2       0.78      0.86      0.82       159
           3       0.88      0.88      0.88       360

    accuracy                           0.86      1200
   macro avg       0.84      0.84      0.84      1200
weighted avg       0.86      0.86      0.86      1200
---------------------------------------------------------
```
#### Model: BERT-base-multilingual-cased (m-BERT-cased)
Parameters: max_len=128, train_batch=16, test_batch=32, epochs = 6, lr=2e-5
```
Accuracy: 0.8466666666666667
Precision: 0.8453902573996248
Recall: 0.8466666666666667
F1-score: 0.8453045673170627
MCC: 0.7747627586951363
---------------------------------------------------------
Class-wise classification report:              
                  precision    recall  f1-score   support

           0       0.87      0.91      0.89       524
           1       0.79      0.69      0.73       157
           2       0.82      0.84      0.83       159
           3       0.85      0.83      0.84       360

    accuracy                           0.85      1200
   macro avg       0.83      0.82      0.82      1200
weighted avg       0.85      0.85      0.85      1200
---------------------------------------------------------
```
#### Model: BERT-base-multilingual-uncased (mBERT-uncased)
Parameters: max_len=128, train_batch=16, test_batch=32, epochs = 6, lr=5e-5
```
Accuracy: 0.8591666666666666
Precision: 0.8603051469756884
Recall: 0.8591666666666666
F1-score: 0.8592904908839932
MCC: 0.7952430871914298
---------------------------------------------------------
Class-wise classification report:             
               precision    recall  f1-score   support

           0       0.90      0.90      0.90       524
           1       0.74      0.72      0.73       157
           2       0.79      0.88      0.83       159
           3       0.89      0.86      0.87       360

    accuracy                           0.86      1200
   macro avg       0.83      0.84      0.83      1200
weighted avg       0.86      0.86      0.86      1200
---------------------------------------------------------
```
#### Model: XLM-RoBERTa
Parameters: max_len=128, train_batch=16, test_batch=32, epochs = 5 lr=2e-5
```
Accuracy: 0.8675
Precision: 0.8696318194488171
Recall: 0.8675
F1-score: 0.867557867752591
MCC: 0.8080296942576948
---------------------------------------------------------
Class-wise classification report:               
               precision    recall  f1-score   support

         0       0.93      0.90      0.91       524
         1       0.79      0.72      0.75       157
         2       0.77      0.90      0.83       159
         3       0.87      0.88      0.87       360

    accuracy                           0.87      1200
   macro avg     0.84      0.85      0.84      1200
weighted avg     0.87      0.87      0.87      1200
---------------------------------------------------------
```
#### Ensemble prediction
Considered top-3 models: RoBERTa, Bangla-BERT-base, and mBERT-uncased
```
Accuracy: 0.8766666666666667
Precision: 0.8775601042666743
Recall: 0.8766666666666667
F1-score: 0.8763746343903924
MCC: 0.8205216999232595
---------------------------------------------------------
Class-wise classification report:
               precision    recall  f1-score   support

           0       0.91      0.90      0.91       524
           1       0.82      0.74      0.78       157
           2       0.79      0.90      0.84       159
           3       0.89      0.89      0.89       360

    accuracy                           0.88      1200
   macro avg       0.85      0.86      0.85      1200
weighted avg       0.88      0.88      0.88      1200
----------------------------------------------------------
```
