---
title: FinalExamPartA
emoji: 🏢
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Final Data Science Exam Part A [Computer Version]
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Predictive Model for H2 and Char Yield
This project provides an interactive web application for predicting H2 (wt.%) and Char yield (wt.%) using machine learning models based on a dataset of chemical compositions and physical properties. The app integrates with Gradio to offer a simple, user-friendly interface for making predictions. It leverages a variety of machine learning models trained on the dataset, such as Random Forest and XGBoost, and provides various performance metrics to assess the accuracy of the predictions.

## Features
1. Exploratory Data Analysis (EDA):
Data Cleaning:
Remove duplicates
Handle missing values (by imputing with median)
Detect and address outliers
Correlation Analysis:
Visualize relationships between features using heatmaps or correlation plots.
Optional PCA:
## Perform Principal Component Analysis (PCA) to reduce dimensionality and visualize the data in 2D space.
2. Machine Learning Models:
The app applies four machine learning models to predict the following target variables:
H2 (wt.%)
Char yield (wt.%)
Models include:
Random Forest Regressor
XGBoost
Support Vector Regression (SVR)
Linear Regression
Model evaluation using various metrics:
R² (Coefficient of Determination)
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
You can evaluate model performance on both the training and test datasets.
## Hyperparameter tuning can be performed to optimize model performance.
3. Partial Dependence Plots (PDP):
Generate a Partial Dependence Plot (PDP) to visualize how a specific feature affects the target variable while holding other features constant.
Input Variables
The model requires the following input variables:

C (%): Carbon content percentage in the sample.
H (%): Hydrogen content percentage in the sample.
N (%): Nitrogen content percentage in the sample.
O (%): Oxygen content percentage in the sample.
S (%): Sulfur content percentage in the sample.
VM (%): Volatile Matter percentage.
Ash (%): Ash content percentage.
FC (%): Fixed Carbon percentage.
Moisture (%): Moisture content percentage.
T (°C): Temperature in degrees Celsius at which the data was collected.
OC (%): Organic Carbon percentage.
SBR: A specific feature related to the dataset, potentially indicating a physical property or chemical compound.
These input features will be entered by the user in the Gradio interface when making predictions.

## Output Variables
The model predicts the following output variables:

H2 (wt.%): Hydrogen content in weight percent. This is a key output that the model predicts based on the input features.
Char yield (wt.%): The char yield percentage, representing the portion of the sample that remains after heating (which can be useful for combustion or other chemical processes).
Once you enter the required input features, the model will predict these two variables for your sample.

## Installation
To run the app locally or deploy it to Hugging Face, follow the instructions below.

## Prerequisites
Ensure you have Python 3.6 or higher installed.

1. Clone the repository or download the project files:
bash
Copy code
git clone <repo_url>
cd <project_directory>
2. Create and activate a virtual environment (optional but recommended):
bash
Copy code
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
3. Install the necessary dependencies:
bash
Copy code
pip install -r requirements.txt
The requirements.txt file contains all the dependencies required to run this app.

## How to Use the App
1. Running the App Locally
After installing the dependencies, you can run the app locally with the following command:

bash
Copy code
python app.py
This will start a local web server, and the Gradio interface will open in your default browser, allowing you to interact with the app.

2. Upload Dataset
Upload an Excel file containing the data, which should include the following columns (example):
C (%), H (%), N (%), O (%), S (%)
VM (%), Ash (%), FC (%), Moisture (%), T (°C), OC (%), SBR
The dataset will be automatically processed to remove duplicates and handle missing values.
3. Enter Input Features
Input the required values for the following features:
C (%)
H (%)
N (%)
O (%)
S (%)
VM (%)
Ash (%)
FC (%)
Moisture (%)
T (°C)
OC (%)
SBR
4. Get Predictions
After entering the input values, click Submit to get the following predictions:
Predicted H2 (wt.%)
Predicted Char yield (wt.%)
5. Model Evaluation (Optional)
The app will provide model evaluation metrics for both training and test datasets, including R², RMSE, and MAE.
6. Visualizations
Correlation Heatmap: Visualize how the features relate to each other.
Partial Dependence Plot (PDP): Analyze the impact of individual features on the target predictions.
Deploying to Hugging Face
You can deploy the app to Hugging Face by following these steps:

Push your code to a GitHub repository.
Connect your repository to Hugging Face using the Hugging Face Spaces platform.
Hugging Face will automatically install the required dependencies from the requirements.txt and deploy the Gradio interface for public access.
To deploy to Hugging Face:
Go to your Hugging Face account and create a new Space.
Select Gradio as the framework.
Link to your GitHub repository or upload your files directly.
Hugging Face will take care of the deployment process and provide you with a public URL for the app.

## Dataset Format
The dataset should contain the following columns for input features:

C (%): Carbon content percentage
H (%): Hydrogen content percentage
N (%): Nitrogen content percentage
O (%): Oxygen content percentage
S (%): Sulfur content percentage
VM (%): Volatile Matter percentage
Ash (%): Ash content percentage
FC (%): Fixed Carbon percentage
Moisture (%): Moisture content percentage
T (°C): Temperature (in °C)
OC (%): Organic Carbon percentage
SBR: A specific feature related to the dataset (e.g., a physical property or chemical compound)
The model will predict the following target variables:

H2 (wt.%): Hydrogen content in weight percent
Char yield (wt.%): Char yield in weight percent
Model Performance Evaluation
The app evaluates the model's performance using the following metrics:

R² (Coefficient of Determination): Measures how well the model's predictions match the actual data.
RMSE (Root Mean Squared Error): Measures the average magnitude of error in the predictions.
MAE (Mean Absolute Error): Measures the average of the absolute errors between predictions and actual values.
Requirements
This project requires the following Python packages, which are listed in the requirements.txt file:

txt
Copy code
gradio==3.30.0

pandas==1.5.3

scikit-learn==1.2.2

openpyxl==3.0.10

seaborn==0.11.2

matplotlib==3.7.0

xgboost==1.7.6

numpy==1.24.2

To install all dependencies, simply run:

bash
Copy code
pip install -r requirements.txt