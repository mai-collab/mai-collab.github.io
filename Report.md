<h1> Using Chemistry Composition for Glass Classification </h1> 

Using collected chemical composition evidence, the performance of key supervised classification models in predicting the right glass types were analyzed.
<p align="center">
<img src="https://github.com/user-attachments/assets/280d49d1-5c5b-409e-bbb6-b419e3f9592b" width = "500"> </p>


<h6 align="center"> Figure 1: An image of shattered glass that can be collected for analysis </h6>

* * *

<h2> Introduction </h2>

Forensic science is an interdisciplinary field that is essential to deciphering the who, what, where, when, and why of a crime committed. In order to do so, crime scene investigators collect and analyze the remaining evidence of a crime scene in hopes of creating a visual of what happened at the scene. Glass is a commonly left over piece of evidence as a result of breaking in, collateral damage of a physical altercation, etc. The type of glass is helpful in reconstructing what exactly happened since glass types have unique fracturing patterns. Eventually, the glass classification can potentially link a suspect to a crime scene, but caution must be taken to avoid misidentifying suspects.

Manual analysis of glass types can be time consuming, subject to human error, and knowledge intensive. Glass classification can utilize machine learning models to speed up the process, standardize analysis, learn from itself, and update itself according to new glass types (if manufactured) -- all with minimal effort from investigators. Crime scene investigators have multitudes of evidence to process, thus training models with glass classification can aid in lessening the workload and allowing them to focus their efforts on components that require manual analysis. 

A glass type database allows a supervised model to be trained to predict glass types using their oxide content. I will train key machine learning models: decision tree classification, random forest classification, support vector classification (SVC), K-nearest neighbor (K-NN), and logistic regression (LR), to assess the best model to implement in glass prediction. 

Upon training and testing, the random forest classifier was the best model due to its highest F1-score of 0.81.

<p align="center">
<img src="https://github.com/user-attachments/assets/d58e9542-e7c4-45d4-adb4-9a521e3c0671" width = "200"> </p>

<h6 align="center"> Figure 2: This is Dexter who is a prominent fictional forensic scientist whose colleagues could use this machine learning project in their work (He is actually a forensic blood spatter analysis, so he wouldn't use this exact project) </h6>

* * *

<h2> Data </h2>
A database on the UC Irvine Machine Learning Repository allows for these models to be trained to predict glass types.

Data: 
Features: Refractive Index, Na, Mg, Al, Si, K, Ca, Ba, Fe
Glass types:

1: Building Windows Float processed (70 instances)

2: Building Windows non Float processed (76 instances)

3: Vehicle windows Float processed (17 instances)

4: Containers (13 instances)

5: Tableware (9 instances)

6: Headlamps (29 instances)

Preprocessing: Remove ID column and replace 1, 2, 3, 5, 6, 7 glass types with 1, 2, 3, 4, 5, 6 (replace 4) because there is no data for glass type 4. 

Decision Tree Classifier Parameters: Setting a maximum depth will prevent the tree from becoming too deep, which leads to overfitting. It was experimentally found at 10 to be the best balance between overfitting and underfitting the training data.

```python
Classifier_Mg = DecisionTreeClassifier(max_depth = 10, random_state=42)
Classifier_Mg.fit(X_train_new, y_train)
```

Random Forest Classifier Parameters: n_estimators controls the amount of decision trees used in creating the random forest model. This number was experimentally found to be optimal at around 1000.

```python
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(X_train_new, y_train)
```
Also, 
```
StandardScaler
```
was used for LR and SVC model training in order to centralize the data around 0 and set a unit variance of 1. This ensures equal contribution from each feature in predictions.

Plots of data:  
<p align="center">
<img src= "https://github.com/user-attachments/assets/9424d021-0397-4d75-9f3d-97e338f2a551" width = "300"> <img src= "https://github.com/user-attachments/assets/b58250d9-8447-4ce1-85c0-3d71980d225c" width = "300"> <img src= "https://github.com/user-attachments/assets/03cce8f2-a8a3-4774-9606-29ff029e362a" width = "300"> <img src= "https://github.com/user-attachments/assets/8294d6b9-5c1a-4c49-a913-8e5a42f97410" width = "300"> <img src= "https://github.com/user-attachments/assets/50b83126-067c-4b96-a0b5-662dc0b8ad78" width = "300"> <img src= "https://github.com/user-attachments/assets/c59e411a-d5ac-4990-a7ec-86108ad12f5d" width = "300"> <img src= "https://github.com/user-attachments/assets/5a4b5a62-7916-4b1f-b04b-86f7a2febc0c" width = "300"> <img src= "https://github.com/user-attachments/assets/0d5341e8-c976-4197-9314-d8dce9af3045" width = "300"> <img src= "https://github.com/user-attachments/assets/297b3561-8904-45fb-a59d-ab11df724483" width = "300"> </p>

<h6 align="center"> Figure 3: Histogram of all nine features </h6>

<p align="center">
<img src= "https://github.com/user-attachments/assets/388f085a-348b-4da1-91e8-a47e3429e2fb" width = "500"> </p>

<h6 align="center"> Figure 4: Correlation Matrix of dataset </h6>

* * *

<h2> Modelling </h2>
Classification Models tried:
All are supervised as the dataset includes data for both input and corresponding outputs. Thus, supervised models can maximize data usage. 

Decision Tree Classifier -- Splits data continuously based on feature values to arrive at a class prediction.

Random Forest Classifier -- Multiple decision trees trained using bootstrapped data which are then averaged to make a prediction.

Logistic Regression (LR) -- Mainly used for binary outputs because it predicts how likely an input points to one of two outputs, rather than predicting the class itself (incorporated this out of curiosity though it does not obviously apply to this dataset of multiple classes).

Support Vector Classifier (SVC) -- Finds the decision boundary that best separates classes by maximizing the margin between the support vectors of each class, then determines which side of the boundary it falls on.

K-Nearest Neighbor (K-NN) -- Classifies using the majority class of the nearest neighbor in the feature space.

**Goal**: Find the best model that can predict glass types based on chemical composition.

**Secondary goal**: Can the features be minimized to just Mg to allow for wider application of the dataset?

* * *

<h2> Results </h2>
Decision Tree Classifier -- Precision: 0.80, Recall: 0.77, F1-score: 0.75
Most important feature: Mg

Random Forest Classifier -- Precision: 0.84, Recall: 0.81, F1-score: 0.81
Most important feature: Mg

Logistic Regression -- Precision: 0.69, Recall: 0.72, F1-score: 0.69

Support Vector Classifier -- Precision: 0.68, Recall: 0.72, F1-score: 0.69

K-Nearest Neighbor -- Precision: 0.62, Recall: 0.63, F1-score: 0.59

<p align="center">
<img src= "https://github.com/user-attachments/assets/4c732129-682e-4a9e-a8eb-7049b482d1c5" width = "450"> <img src= "https://github.com/user-attachments/assets/a0ec93a3-ddea-4bed-83a2-487596f8e8bc" width = "450"> <img src= "https://github.com/user-attachments/assets/37c4b8ac-059a-4e9e-8497-814b88725c5a" width = "450"> <img src= "https://github.com/user-attachments/assets/53bcbc92-3925-4176-83c4-2835b355207e" width = "450"> <img src= "https://github.com/user-attachments/assets/042e07d7-4711-46b7-ade1-eaf6dc81485c" width = "450"> </p>

<h6 align="center"> Figure 5: Confusion matrix (CM) for all five models </h6>

**Training models with only Mg as the feature**

Decision Tree Classifier -- Precision: 0.44, Recall: 0.35, F1-score: 0.36

Random Forest Classifier -- Precision: 0.44, Recall: 0.33, F1-score: 0.34

Logistic Regression -- Precision: 0.34, Recall: 0.37, F1-score: 0.35

Support Vector Classifier -- Precision: 0.39, Recall: 0.30, F1-score: 0.23

K-Nearest Neighbor -- Precision: 0.42, Recall: 0.42, F1-score: 0.40
<p align="center">
<img src= "https://github.com/user-attachments/assets/8579a2f6-f75e-4cd6-b098-39f6b5f13146" width = "450"> <img src= "https://github.com/user-attachments/assets/80ca6497-90a7-427b-8e37-25b3697fdae2" width = "450"> <img src= "https://github.com/user-attachments/assets/848dfad7-8893-4a34-a750-51de39bfa488" width = "450"> <img src= "https://github.com/user-attachments/assets/fd1e11e7-3587-4bf5-89b2-1d2fcf0cba09" width = "450"> <img src= "https://github.com/user-attachments/assets/d82a5e0e-d617-4a18-8c3c-bcc6ef02a02f" width = "450"> </p>
  
<h6 align="center"> Figure 6: Confusion matrix (CM) for all 5 models trained only with Magnesium (Mg) </h6>

* * *

<h2> Discussion </h2>
The database provided is imbalanced with the lowest class (5) having 9 instances and the highest class (2) having 76 instances. Thus, the predictions by the models are focused on the classes with higher instances in the dataset: classes 1, 2, and 6. In order to minimize the effects of the imbalanced dataset, the weighted average was used in the metric analyses of each model which takes into account the contributions from each class predicted. 

Precision measures the amount of true positives out of all predicted positives by the model, and a high value indicates success at avoiding false positives. Recall measures the true positives out of all actual positives, and a high value indicates success at avoiding false negatives. The f1-score combines both precision and recall into one metric. For this glass classification goal, both precision and recall is important as false positives can misidentify a suspect while a false negative will overlook potentially critical evidence. Thus, the f1-score will be prioritized in this analysis. 

The best model at predicting glass types was the random forest classification model, which had an f1-score of 0.81. Its outperformance relative to the decision tree is expected as the random forest takes into account multiple decision trees that were trained on bootstrapped data. Thus, it is less prone to any training data bias. As shown in the CMs in figure 5, the decision tree and random forest classifiers produced the relatively clearest diagonal line indicating a higher concentration of correct predictions as opposed to the remaining models which failed to produce a clear diagonal. Decision trees can evaluate the importance of different features in predicting the glass type and can run multi-class classifications which can support its ability to best predict glass types. In addition, it is widely accepted that tree based machine learning algorithms are statistically superior in predicting output values in comparison to SVC, LR, and K-NN models (Uddin and Lu, 2024 [^1]). 

Though tree based algorithms can prioritize feature importance and predict classes successfully, they have their limitations when it comes to overfitting, large computational cost, and bias to imbalanced data. The trees aim to split at and classify every datapoint in the training phase, which can grow the tree excessively and lead to overfitting the data. Setting a maximum depth and n_estimators value can help to alleviate this issue by controlling the amount of splits in the trees and the number of trees itself, respectively. In addition, the trees can begin to ignore minority instance classes when making tree splits as there is not enough data in those classes. Also, a high computational cost can slow the training time and increase hardware costs. A random forest mitigates some of these issues to an extent by averaging out the results from multiple decision trees, however the limitations should be noted if applying this model to other evidence classification tasks. 

K-NN was the worst at prediction, with an f1-score of 0.59, likely due to its sensitivity to high-dimensional datasets and inability to learn complex decision boundaries. Since K-NN works by essentially finding the nearest data point to a given input, its failure to predict this multi-feature problem is expected. It can be difficult and complex to plot multiple features in order to find the nearest cluster. The simplicity of this model limits its ability to take into account all nine features in making a prediction. 

In addition, the effects of lowering the number of features to one was considered in order to train faster, more efficient, and easier to interpret models. The correlation matrix in figure 4 shows that Mg has the highest correlation with the glass column, -0.73. The data was preprocessed to only include Mg as the input and each model was retrained. Unfortunately, the highest f1-score produced was 0.40 by the K-NN model. The K-NN model thrives with less features as it finds the nearest input data point to make a majority prediction. This performance is expected as the decision boundary is significantly simpler and linear with only one feature. Also, the data points are likely closer together with just one feature, so the model can better find neighbors that correctly predict the class. Nonetheless, the model was not great at predicting the glass types. As evidenced by figure 6, all 5 models failed to produce CMs that were comparable to the models trained on the entire dataset, let alone produce a diagonal line. It is discouraged to utilize a single feature in predicting glass types, though its wide application could have been useful.

* * *

<h2> Conclusion </h2>

Testing five supervised learning, classification models on a glass classification task with nine features showed that the random forest classifier outperforms the decision tree classifier, logistic regressor, support vector classifier, and K-nearest neighbor models. Many aspects of crime scene investigation incorporate the classification of an unknown object, thus random forest classifiers can be trained using chemical composition to automate some aspects of classifying unknowns in the crime scene. This can help reconstruct what happened during the crime and aid in linking suspects. Then, time can be dedicated to evidence analyses that require manual attention. 

In addition, it was found that glass classification is not feasible with just magnesium as the feature. Though it would make model interpretation simpler, reduce computational cost, and allow for quicker evidence collection, the best model, K-NN, performed worse than the worst model, also K-NN, that was trained on the entire dataset.

With the information on decision tree algorithm superiority, other evidence classification tasks can be automated, such as blood spatter analysis, for Dexter in figure 2. Essentialy, any task involving chemical composition to classify the unknown can benefit from a tree based machine learning approach, notably random forest classifier models.

<h2> References </h2>

[^1]: Uddin, S., & Lu, H. (2024). Confirming the statistically significant superiority of tree-based machine learning algorithms over their counterparts for tabular data. PloS one, 19(4), e0301541. https://doi.org/10.1371/journal.pone.0301541
