# Healthcare-Stroke-Prediction-ML-Model

This project focuses on predicting the likelihood of a stroke occurrence based on patient medical data. Using various machine learning models, the goal is to build a robust prediction model that can assist healthcare providers in identifying individuals at risk of having a stroke.

## Table of Contents
* Project Overview
* Dataset
* Project Workflow
* Installation and Requirements
* Models and Evaluation
* Results
* Conclusion
* Usage

## Project Overview
In this project, I used the healthcare-dataset-stroke-data.csv to predict the likelihood of stroke based on a variety of patient data such as age, gender, hypertension, heart disease, and others. The dataset includes several categorical and numerical features. Various machine learning models were employed to classify whether a patient is likely to experience a stroke or not.

## Dataset
The dataset used for this project is healthcare-dataset-stroke-data.csv. It contains 5110 rows and 12 columns with the following key features:

* id: Unique identifier for each patient
* gender: Gender of the patient
* age: Age of the patient
* hypertension: Whether the patient has hypertension
* heart_disease: Whether the patient has heart disease
* ever_married: Whether the patient has been married
* work_type: Type of work the patient does
* residence_type: Whether the patient lives in an urban or rural area
* avg_glucose_level: The average glucose level in blood
* bmi: Body Mass Index
* smoking_status: Smoking habits of the patient
* stroke: Binary variable indicating whether the patient has experienced a stroke
## Project Workflow

 1. Data Preprocessing:
* Data cleaning, handling missing values, and dealing with outliers.
* Oversampling to address class imbalance.
* Standardization of numerical features using StandardScaler.

2. Feature Selection:
* Selected the most important 8 features using a feature selection technique.

3. Model Selection:
* Trained various classification models: LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier, SVC, and GaussianNB.

4. Hyperparameter Tuning:
* Used GridSearchCV to tune hyperparameters for optimal model performance.

5. Evaluation:
* Models were evaluated using metrics such as accuracy and ROC AUC score.
Installation and Requirements

6. Models and Evaluation
We explored the following models:
* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier
* K-Nearest Neighbors
* Support Vector Classifier (SVC)
* Gaussian Naive Bayes (GaussianNB)
* The best performing model was RandomForestClassifier with an accuracy of 99.43% and an ROC AUC score of 1.0.

7. Results
The Random Forest Classifier proved to be the most effective model for predicting stroke, outperforming other models in accuracy and ROC AUC score. The function to predict stroke probability was created and tested using various inputs.

8. Conclusion
This project demonstrates the application of machine learning in healthcare by predicting stroke risk using patient data. The Random Forest Classifier was identified as the most reliable model based on its performance metrics. This model, along with the FastAPI implementation, provides an efficient tool for real-world application in stroke prediction.
