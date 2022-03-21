# Tackling Imbalanced Dataset with Stroke Dataset

## What's the point?
The purpose of this exercise is to dive deeper into the techniques and strategies handling of Imbalanced Datasets as a machine learning classification problem. 

## Imbalanced Dataset
An imbalanced dataset is when the distribution of the target classes of either a binary or multi-class dataset are not uniformed. These types of dataset usually show up in problems like disease detection, machine failure detection or fraud detection where the class that we are trying to detect (stroke/heart attack occurences, fraud events and machine failure events) belong to only a minority of the class distribution. This is because events such as fraud, failures or even stroke are rather uncommon. 

## So, what's the big deal about it?
Even though the events that we are trying to detect are uncommon, the costs of those events happening and catching us unprepared are usually very, very expensive. For example, the cost of fixing a failed machine is usually more expensive than turning off the machine for predictive maintenance. The cost of having a stroke or heart attack is definitely more expensive than to have an early detection and prevention of those diseases. 

## Dataset Used
We are using the Stroke dataset provided in Kaggle (URL: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset#). A disclaimer to put forth upfront is that the original source of this dataset was not divulged by the Kaggler, hence the validity or the authenticity of the dataset cannot be determined. That being said, the nature of the dataset still well represents the "Imbalanced Classification" type of problem.

## What is covered in this Notebook?
This Notebook starts off **Exploratory Data Analysis (EDA)** of the data where we look at the basic statistical distribution of the data, determining the data quality and getting a basic understanding of how each feature correlates to our target variable (Stroke or No Stroke).

![Distribution of Dataset](https://i.imgur.com/oGgFMdF.png)

_Figure 1: Distribution of Dataset_
<br>
<br>
The dataset is split into `train` and `test` sets. We ensure that the distribution of the imbalanced classes is preserved for both `train` and `test` sets using `StratifiedShuffleSplit`.

``` jupyter
***** Train set distribution *****
0    0.951321
1    0.048679
Name: stroke, dtype: float64

***** Test set distribution *****
0    0.951076
1    0.048924
Name: stroke, dtype: float64
```

We then proceed to the **Data Preparation** stage where we look into different techniques of imputation, feature scaling and categorical enconding. We prepare a pipeline for these data preparation steps using Scikit-Learn's `pipeline`. This `pipeline` is then used as the input to the models that we will evaluate later.

In the **Modeling** stage, we look into different models, starting from Logistic Regression, Random Forest, Support Vector Machine (SVM) and Bagging Classifiers to determine the model as our baseline performance. Main determinant metric used as Recall  scores.

During this time, we also look into how we can **handle the imbalanced classes** by employing techniques such as Random Oversampling, Random Undersampling and Synthetic Minority Oversampling Technique (SMOTe). We then evaluate the model performance for these techniques used. Apart from this, we also explore using different variations of models such as Balanced Random Forest, Balanced Bagging Classifier and Penalized SVM.

Once this experiment was completed, we selected "Balanced Random Forest" model as our **basecase model** for finetuning.

In the **finetuning** stage, we employ `RandomSearchCV` followed by `GridSearchCV` to determine the best combinations of hyperparameters for the model

Once the model is finetuned, we perform **threshold sensitivities** to determine which threshold would provide the best model metric performance.

We **test our final model against the test set** that was split at the very beginning.

The supplementary portion of this also includes the **explainability of the model**'s predictions through Random Forest's `feature importance`, where we determine which feature has the highest impact on the prediction of the model.


## Metric Used 
 Selection of metrics for different domains such as disease prediction vs failure prediction is another topic of research and discussion itself, but for this exercise we are using **Recall** score as our main primary metric as we are aiming to have a higher number of stroke prediction. Since the cost of not detecting stroke when there is is higher, we would like to be able to detect more stroke patients than missing them out.

## Techniques employed to tackle the imbalanced distribution
* **Sampling techniques** (Random Oversampling, Random Undersampling SMOTe)
    - Because of the imbalance in the classes (majority "No Stroke"), the vanilla models such as `LogisticRegression` and even `RandomForestClassifier` out of the box will be completely biased to the majority class. 
    - We can alleviate this by creating "Synthetic" minority dataset based on its distribution 
        - `SMOTe`, which stands for Synthetic Minority Over-sampling Technique, will take a sample from the minority dataset, find its k-nearest neighbors and obtain a vector between the sample and one of the k neighbors and multiply that with a random number between 0 and 1.
        - `RandomOverSampler` and `RandomUnderSampler` will randomly (and naively) oversample or undersample the minority or majority class, respectively.
        - As we are also employing cross validation technique during our training stage, much care needs to be taken to avoid data leakage, where during training time, our model happens to be seeing data points that are similar to points that were supposedly kept away for the cross-validation stage. To avoid this, the `imblearn` package also provides its version of `Pipeline` that allows us to only resample only within the each fold during the cross validation stage (as opposed to resampling the whole training data).  
* **Customized models**
    - `imblearn` package also provides models that are customized from the vanilla `sklearn` models such as `RandomForestClassifier` and `BaggingCLassifier`.
    - Balanced Random Forest Classiifer model:
        - Each tree of the balanced forest will be provided a balanced bootstrap sample (by randomly under-sampling each boostrap sample). This will help the model to not "look too much" at the majority class at each bootstrap.
    - Balanced Bagging Classifier:
        - Each bootstrap sample will be further resampled to achieve the `sampling_strategy` desired.

* **Moving Threshold**

    - As the models provide probabilities of the predictions, we can actually tune the threshold in order optimize our performance based on the metrics that we have chosen. the threshold ptimal threshold when converting probabilities to crisp class labels for imbalanced classification.
    
    
## Model Explainability: Feature Importance

As we are using a Random Forest model, we are able to determine which of the independent variables / features are playing the biggest predictive role in the model.

![Feature Importance](https://i.imgur.com/hLGIVPT.png)

_Feature Importance_

## References: 

1. https://scikit-learn.org/
2. https://imbalanced-learn.org/stable/
3. https://elitedatascience.com/imbalanced-classes
4. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
5. https://en.wikipedia.org/wiki/F-score
6. https://www.publichealth.columbia.edu/research/population-health-methods/evaluating-risk-prediction-roc-curves
7. https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
8. https://en.wikipedia.org/wiki/Precision_and_recall
9. Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow by Aurélien Géron

