# Credit_Risk_Analysis

## Analysis Overview
We were tasked with utilizing Python to build and evaluate several machine learning models to predict credit risk, following the procedures below:
- Oversampling the data using the RandomOverSampler algorithm.
- Undersampling the data using the ClusterCentroids algorithm.
- Using a combinatorial approach of over and undersampling using the SMOTEENN algorithm. 
- Comparing two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier.

We evaluated the performance of these models and made a recommendation on whether they should be used to predict credit risk.

There are four ways to check if the predications are right or wrong:
- TN/True Negative: the case was negative and predicted negative
- TP/True Positive: the case was positive and predicted positive
- FN/False Negative: the case was positive but predicted negative
- FP/Flase Positive: the case was negative but predicted positive<br>

Precision: Accuracy of positive predictions.
Recall: Fraction of positives that were correctly identified.
F1 score: A weighted harmonic mean of precision and recall.

## Resources
- Data Sources: LoanStats_2019Q1.csv
- Software: Python, Anaconda Navigator, Conda, Jupyter Noteook.

## Results (Balanced Accuracy Scores, Confusion Matrixes and Imbalanced Classification Reports)

### RandomOverSampler model
<p align="center"><img width="600" alt="Screen Shot 2022-05-10 at 1 51 01 PM" src="https://user-images.githubusercontent.com/96352751/167691542-76f5bbe0-9400-4f01-8783-4c81683fe125.png">

<img width="600" alt="Screen Shot 2022-05-10 at 2 00 55 PM" src="https://user-images.githubusercontent.com/96352751/167693166-d93b8871-f4ec-4cc0-98c0-583d4fba2ab7.png">
  
<img width="600" alt="Screen Shot 2022-05-10 at 1 55 30 PM" src="https://user-images.githubusercontent.com/96352751/167692264-4c5c3a77-ef3b-4c08-b677-fb8e5ecbfb9f.png">
 </p>

The balance accuracy score is 64%. The precision column ("pre") for both classes is in the 99th percentile or higher, whereas the recall ("rec") column for both classes is 0.69 and 0.59. This model does not appear to be overly strong in predicting the risk classes.

### SMOTE model
<p align="center"><img width="600" alt="Screen Shot 2022-05-10 at 2 39 35 PM" src="https://user-images.githubusercontent.com/96352751/167699502-ba9e49f7-c1be-4a3a-9275-d31a2572a206.png">
  
<img width="600" alt="Screen Shot 2022-05-10 at 2 42 06 PM" src="https://user-images.githubusercontent.com/96352751/167699796-23506b33-a059-44fb-9bf9-181aedc993d3.png">

<img width="600" alt="Screen Shot 2022-05-10 at 2 43 43 PM" src="https://user-images.githubusercontent.com/96352751/167700084-ce83a76a-e0ad-45eb-9fc9-b165d48a84ee.png">
</p>

The results are similar to the previous model, where the balanced accuracy score is agains 64%. Both classes are in the 99th percentile, and the recall for both classes is in the same range, 0.61 and 0.69. The oversampling technique is slightly stronger then the naive random sampling model.

### ClusterCentroids model
<p align="center">
<img width="600" alt="Screen Shot 2022-05-10 at 2 57 56 PM" src="https://user-images.githubusercontent.com/96352751/167702429-8b2051be-0100-485e-baa6-4e6a9305a6b6.png">
  
<img width="600" alt="Screen Shot 2022-05-10 at 2 59 27 PM" src="https://user-images.githubusercontent.com/96352751/167702633-7948d2ac-5b7d-4baa-8cee-6860fa37bd4a.png">

<img width="600" alt="Screen Shot 2022-05-10 at 3 00 37 PM" src="https://user-images.githubusercontent.com/96352751/167703852-98ba07d2-55cb-43f8-9b9f-220d4c99a30c.png">
</p>

The balanced accuracy score is 57%. This technique leads to similar precision results but the recall is worse.

### SMOTEENN model
<p align="center"><img width="600" alt="Screen Shot 2022-05-10 at 3 20 17 PM" src="https://user-images.githubusercontent.com/96352751/167705973-eb2ca565-acd9-4894-a30b-138db1cb9946.png">
  
<img width="600" alt="Screen Shot 2022-05-10 at 3 21 51 PM" src="https://user-images.githubusercontent.com/96352751/167706124-018664d8-0d00-4b6e-bdfb-20bd4ea29b4e.png">

<img width="600" alt="Screen Shot 2022-05-10 at 3 22 54 PM" src="https://user-images.githubusercontent.com/96352751/167706310-2a84f867-e32f-455c-a2c8-dafeea81b66e.png">

The balanced accuracy score is 62%. The results for the SMOTEENN model is similar that of the SMOTE oversampling with recall at 87%.
  
### BalancedRandomForestClassifier model

<p align="center"><img width="600" alt="Screen Shot 2022-05-10 at 3 34 44 PM" src="https://user-images.githubusercontent.com/96352751/167708359-ecca9029-9f06-45de-84bb-5722b7961ff7.png">
  
<img width="600" alt="Screen Shot 2022-05-10 at 3 35 04 PM" src="https://user-images.githubusercontent.com/96352751/167708421-4d5dd93a-2b30-4b03-a32f-941a8b290ff3.png">

<img width="600" alt="Screen Shot 2022-05-10 at 3 35 26 PM" src="https://user-images.githubusercontent.com/96352751/167708445-f020a837-c589-41b4-8141-bd2224504fb4.png">
</p>

The balanced accuracy score improved to 87%. The high_risk precision is 3%. 

### EasyEnsembleClassifier model
<p align="center"><img width="600" alt="Screen Shot 2022-05-10 at 3 44 12 PM" src="https://user-images.githubusercontent.com/96352751/167709740-a7f90c19-58b7-4ca3-8ac5-ddf86863f199.png">

<img width="600" alt="Screen Shot 2022-05-10 at 3 44 41 PM" src="https://user-images.githubusercontent.com/96352751/167709775-44c98d0e-0548-4e3c-a33b-0653ba63ac3e.png">

<img width="600" alt="Screen Shot 2022-05-10 at 3 45 12 PM" src="https://user-images.githubusercontent.com/96352751/167709808-2892fcf5-c7f9-4ca6-a1d3-3db6c8c46972.png">
