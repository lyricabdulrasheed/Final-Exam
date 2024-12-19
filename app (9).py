import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.inspection import partial_dependence
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Function for exploratory data analysis (EDA)
def plot_correlation(data):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# PCA analysis
def perform_pca(data, features):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data[features])
    plt.scatter(principal_components[:, 0], principal_components[:, 1])
    plt.title("PCA of the Dataset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

# Data cleaning and preprocessing
def preprocess_data(data):
    # Clean column names
    data.columns = data.columns.str.strip()
    
    # Remove duplicates
    data.drop_duplicates(inplace=True)
    
    # Handle missing values (fill with median)
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    return data

# Train machine learning models
def train_models(data, features, targets):
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "KNN Regressor": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    best_models = {}
    results = {}
    
    for target in targets:
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Model Evaluation Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            }
            
            best_models[model_name] = model
            
    return best_models, results

# Plot Partial Dependence for a given model
def plot_partial_dependence_plot(model, X):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use the new `partial_dependence` function for plotting
    pdp = partial_dependence(model, X, features=[0, 1, 2])  # For the first 3 features
    # Plot the partial dependence
    ax.plot(pdp['values'][0], pdp['average'][0])
    ax.set_xlabel('Feature Values')
    ax.set_ylabel('Partial Dependence')
    ax.set_title('Partial Dependence for Feature 1')
    plt.show()

# Prediction function for Gradio interface
def predict(models, C, H, N, O, S, VM, Ash, FC, Moisture, T, OC, SBR):
    try:
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            "C (%)": [C],
            "H (%)": [H],
            "N (%)": [N],
            "O (%)": [O],
            "S (%)": [S],
            "VM (%)": [VM],
            "Ash (%)": [Ash],
            "FC (%)": [FC],
            "Moisture (%)": [Moisture],
            "T (°C)": [T],
            "OC (%)": [OC],
            "SBR": [SBR]
        })
        
        results = {}
        for model_name, model in models.items():
            predictions = model.predict(input_data)
            results[model_name] = predictions[0]
        
        return results
    
    except Exception as e:
        return f"Error: {str(e)}", ""

# Gradio interface setup
def create_gradio_interface(models):
    inputs = [
        gr.Number(label="C (%)", value=10),
        gr.Number(label="H (%)", value=5),
        gr.Number(label="N (%)", value=2),
        gr.Number(label="O (%)", value=10),
        gr.Number(label="S (%)", value=5),
        gr.Number(label="VM (%)", value=20),
        gr.Number(label="Ash (%)", value=3),
        gr.Number(label="FC (%)", value=8),
        gr.Number(label="Moisture (%)", value=2),
        gr.Number(label="T (°C)", value=25),
        gr.Number(label="OC (%)", value=1),
        gr.Number(label="SBR", value=1),
    ]
    
    outputs = [gr.Textbox(label=model_name) for model_name in models.keys()]
    
    iface = gr.Interface(
        fn=lambda *args: predict(models, *args),
        inputs=inputs,
        outputs=outputs,
        title="Predict Output for Multiple Models",
        description="Enter values for features and get predictions from various models."
    )
    
    iface.launch(share=True)

# Main function to execute
def main(file):
    # Read the dataset from uploaded file
    data = pd.read_excel(file.name)  # Use the file's name property in Gradio
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Define features and targets
    features = ['C (%)', 'H (%)', 'N (%)', 'O (%)', 'S (%)', 'VM (%)', 'Ash (%)', 'FC (%)', 'Moisture (%)', 'T (°C)', 'OC (%)', 'SBR']
    targets = ['H2 (wt.%)', 'Char yield (wt.%)']
    
    # Perform EDA (optional)
    plot_correlation(data)  # Show correlation heatmap
    perform_pca(data, features)  # Perform PCA for dimensionality reduction
    
    # Train models
    models, results = train_models(data, features, targets)
    
    # Print evaluation results
    for model_name, result in results.items():
        print(f"Results for {model_name}:")
        print(f"  R²: {result['R²']:.2f}")
        print(f"  RMSE: {result['RMSE']:.2f}")
        print(f"  MAE: {result['MAE']:.2f}")
    
    # Plot Partial Dependence for one model (e.g., Random Forest)
    plot_partial_dependence_plot(models['Random Forest'], data[features]) 
    
    # Launch Gradio interface
    create_gradio_interface(models)

# Gradio File Upload Interface
def upload_file(file):
    main(file)

iface = gr.Interface(
    fn=upload_file,
    inputs=gr.File(label="Upload Excel File"),
    outputs="text",
    title="Machine Learning Model Prediction",
    description="Upload an Excel dataset and get predictions from various machine learning models."
)

iface.launch(share=True)
