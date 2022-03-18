# Tackling Imbalanced Dataset with Stroke Dataset

## What's the point?
The purpose of this exercise is to dive deeper into the techniques and strategies handling of Imbalanced Datasets as a machine learning classification problem. 

## Imbalanced Dataset
An imbalanced dataset is when the distribution of the target classes of either a binary or multi-class dataset are not uniformed. These types of dataset usually show up in problems like disease detection, machine failure detection or fraud detection where the class that we are trying to detect (stroke/heart attack occurences, fraud events and machine failure events) belong to only a minority of the class distribution. This is because events such as fraud, failures or even stroke are rather uncommon. 

## So, what's the big deal about it?
Even though the events that we are trying to detect are uncommon, the costs of those events happening and catching us unprepared are usually very, very expensive. For example, the cost of fixing a failed machine is usually more expensive than turning off the machine for predictive maintenance. The cost of having a stroke or heart attack is definitely more expensive than to have an early detection and prevention of those diseases. 

## Dataset Used
We are using the Stroke dataset provided in Kaggle (URL: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset#). A disclaimer to put forth upfront is that the original source of this dataset was not divulged by the Kaggler, hence the validity or the authenticity of the dataset cannot be determined. That being said, the nature of the dataset is still presented to us as an "Imbalanced Classification" problem.

## What is covered in this Notebook?
This Notebook starts off **Exploratory Data Analysis (EDA)** of the data where we look at the basic statistical distribution of the data, determining the data quality and getting a basic understanding of how each feature correlates to our target variable (Stroke or No Stroke).
![image.png](attachment:850ded0c-8705-4f84-8515-cf8b0b2ee10b.png)

The dataset is split into `train` and `test` sets. We ensure that the distribution of the imbalanced classes is preserved for both `train` and `test` sets using `StratifiedShuffleSplit`.

![image.png](attachment:8614346f-d510-4bc9-9874-e55c0de3f9d7.png)

We then proceed to the **Data Preparation** stage where we look into different techniques of imputation, feature scaling and categorical enconding. We prepare a pipeline for these data preparation steps using Scikit-Learn's `pipeline`. This `pipeline` is then used as the input to the models that we will evaluate later.

In the **Modeling** stage, we look into different models, starting from Logistic Regression, Random Forest, Support Vector Machine (SVM) and Bagging Classifiers to determine the model as our baseline performance. Main determinant metric used as Recall  scores.

During this time, we also look into how we can **handle the imbalanced classes** by employing techniques such as Random Oversampling, Random Undersampling and Synthetic Minority Oversampling Technique (SMOTe). We then evaluate the model performance for these techniques used. Apart from this, we also explore using different variations of models such as Balanced Random Forest, Balanced Bagging Classifier and Penalized SVM.

Once this experiment was completed, we selected "Balanced Random Forest" model as our **basecase model** for finetuning.

In the **finetuning** stage, we employ `RandomSearchCV` followed by `GridSearchCV` to determine the best combinations of hyperparameters for the model

Once the model is finetuned, we perform **threshold sensitivities** to determine which threshold would provide the best model metric performance.

We **test our final model against the test set** that was split at the very beginning.

The supplementary portion of this also includes the **explainability of the model**'s predictions through Random Forest's `feature importance`, where we determine which feature has the highest impact on the prediction of the model.


## Metric Used 
As we do not have a reference for this dataset, we are using **Recall** score as our main primary metric as we would prefer to have higher number of stroke prediction. Since the cost of not detecting stroke when there is is higher, we would like to be able to detect more stroke patients than missing them out.

## Techniques employed to tackle the imbalanced distribution
    . Sampling technqiues (Random Oversampling, Random Undersampling SMOTe)
    . Moving threshold
    . Customized models 


```python
#%pip install -U imbalanced-learn seaborn
```


```python
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, accuracy_score, classification_report, fbeta_score, make_scorer

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline, Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from pprint import pprint as pp
import joblib
```

## References: 

1. https://elitedatascience.com/imbalanced-classes
2. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
3. https://en.wikipedia.org/wiki/F-score
4. https://www.publichealth.columbia.edu/research/population-health-methods/evaluating-risk-prediction-roc-curves
5. https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
6. https://en.wikipedia.org/wiki/Precision_and_recall
7. Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow by Aurélien Géron
