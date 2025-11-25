# 📊 Customer Churn Prediction System

## 📝 Overview
This project is a robust Machine Learning application designed to predict customer churn in the banking/telecom sector. It leverages an **XGBoost (Extreme Gradient Boosting)** classifier to identify customers at high risk of leaving the service. The system is deployed via an interactive **Streamlit** dashboard, enabling business users to upload data, visualize insights, and take proactive retention measures.

## 🚀 Features
- **Advanced Modeling**: Utilizes XGBoost, a state-of-the-art gradient boosting algorithm known for high performance and speed.
- **Interactive Dashboard**:
    - **Data Upload**: Support for CSV file uploads for batch predictions.
    - **Churn Visualization**: Interactive Donut charts and histograms to understand churn distribution.
    - **Risk Analysis**: Box plots and bar charts to analyze how features like Credit Score, Age, and Balance impact churn.
    - **High-Risk Filtering**: Dynamic sliders to filter and export lists of high-risk customers.
- **Explainability**: Feature Importance analysis to highlight the top drivers of churn.

## 📂 Project Structure
```
c:/Task_02/
├── app.py                  # Main Streamlit application for the dashboard
├── train_model.py          # Script for data preprocessing, model training, and evaluation
├── best_model.pkl          # Serialized XGBoost model (trained artifact)
├── scaler.pkl              # Serialized StandardScaler for data normalization
├── feature_importance.csv  # CSV containing feature importance scores
├── test_predictions.csv    # Sample dataset with predictions for demo purposes
├── Churn_Modelling.csv     # Raw dataset used for training
└── README.md               # Project documentation
```

## 🧠 Model Details
- **Algorithm**: XGBoost Classifier (`XGBClassifier`)
- **Preprocessing**:
    - **Scaling**: Standard Scaling for numerical features.
    - **Encoding**: One-Hot Encoding for geography, Label Encoding for gender.
- **Performance**: The model is evaluated on Accuracy, Precision, Recall, and ROC-AUC scores.
- **Key Predictors**: The model identifies **Age**, **Number of Products**, and **Balance** as significant indicators of customer churn.

## 🛠️ Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
```

### 2. Running the Dashboard
Launch the Streamlit app:
```bash
streamlit run app.py
```
- **Upload Data**: Use the sidebar to upload a CSV file (schema must match `Churn_Modelling.csv`).
- **Demo Mode**: Click "Use Demo Data" to explore the app with pre-loaded test predictions.

### 3. Retraining the Model
To retrain the model with new data or different hyperparameters:
```bash
python train_model.py
```
This will generate new `best_model.pkl` and `scaler.pkl` files.

## 📊 Insights
- **Demographics**: Older customers tend to have a higher churn risk.
- **Geography**: Regional differences (e.g., Germany) may show higher churn rates due to market conditions.
- **Engagement**: Inactive members are significantly more likely to churn.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests for improvements!
