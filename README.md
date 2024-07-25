# Customer-Churn-Prediction-Model
## Description
This project aims to predict customer churn for a telecom company using various machine learning algorithms. The dataset used in this project is the Telecom Customer Churn dataset. The final model is deployed using Flask to provide an API for making predictions.
## Steps and Methodologies
### Loading the Dataset: 
Import the dataset and understand its structure.
### Exploratory Data Analysis (EDA):
Analyzed the variables using univariate and bivariate analysis and multivariate analysis.
Visualized high churn cases to gain deeper insights.
### Data Preprocessing:
#### 1.Handling missing values
#### 2.Encoding categorical variables
#### 3.Feature scaling
#### 4.Balancing the dataset using SMOTEENN

Feature Selection: Selecting relevant features for model training.
## Machine Learning Models:
#### 1.Logistic Regression: Initial model to understand the baseline performance.

#### 2.Artificial Neural Network (ANN): Built and trained an ANN model using PCA-transformed data.

#### 3.Decision Tree Classifier: Evaluated the performance on original and resampled data.

#### 4.Random Forest Classifier: Applied hyperparameter tuning and evaluated the performance.

### PCA Analysis
Principal Component Analysis (PCA) was used to reduce the dimensionality of the dataset before training the ANN model.

### Hyperparameter Tuning
Applied hyperparameter tuning to find the best parameters.

##Best Performing Model
Out of the four machine learning algorithms used, the best performing model was identified using performance metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.

## Model Evaluation: 
Used various metrics like accuracy, precision, recall, F1-score, and confusion matrix to evaluate model performance.
## Model Deployment: 
Deployed the final model using Flask to create an API endpoint for predictions.


## Deployment with Flask
The final Random Forest model was deployed using Flask to create a web service for making predictions. The Flask app provides an API endpoint that accepts customer data and returns the churn prediction.

