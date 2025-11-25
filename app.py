import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Churn Prediction System", layout="wide")

# Title and Description
st.title("📊 Customer Churn Prediction System")
st.markdown("""
This application predicts the probability of customer churn using a Machine Learning model (XGBoost).
Upload a customer dataset to get predictions and insights.
""")

# Sidebar
st.sidebar.header("User Input Features")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Load Model and Scaler
@st.cache_resource
def load_model():
    model = joblib.load('c:/Task_02/best_model.pkl')
    scaler = joblib.load('c:/Task_02/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    # Data Preprocessing
    st.subheader("1. Data Preview")
    st.dataframe(input_df.head())

    # Preprocess for prediction
    # We need to match the preprocessing steps from training
    # Drop irrelevant columns if they exist
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
    df_processed = input_df.drop([c for c in drop_cols if c in input_df.columns], axis=1)
    
    # Encode Gender
    if 'Gender' in df_processed.columns:
        gender_map = {'Female': 0, 'Male': 1}
        # Handle case where Gender might already be encoded or different case
        if df_processed['Gender'].dtype == 'object':
             df_processed['Gender'] = df_processed['Gender'].map(gender_map)
    
    # One-hot encode Geography
    # We need to ensure we have the same columns as training
    # Geography_Germany, Geography_Spain (Geography_France was dropped as reference if drop_first=True)
    # But wait, get_dummies might create different columns if not all categories are present.
    # A more robust way is to manually create the columns.
    
    if 'Geography' in df_processed.columns:
        df_processed['Geography_Germany'] = (df_processed['Geography'] == 'Germany').astype(int)
        df_processed['Geography_Spain'] = (df_processed['Geography'] == 'Spain').astype(int)
        df_processed = df_processed.drop('Geography', axis=1)
    else:
        # If columns are missing, add them as 0
        if 'Geography_Germany' not in df_processed.columns:
            df_processed['Geography_Germany'] = 0
        if 'Geography_Spain' not in df_processed.columns:
            df_processed['Geography_Spain'] = 0

    # Ensure column order matches scaler/model
    # We need to know the expected columns. 
    # Based on training: CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain
    expected_cols = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain']
    
    # Reorder and fill missing
    for col in expected_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    df_processed = df_processed[expected_cols]
    
    # Scale
    X_scaled = scaler.transform(df_processed)
    
    # Predict
    prediction = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Add predictions to original dataframe
    input_df['Predicted_Churn'] = prediction
    input_df['Churn_Probability'] = prediction_proba
    
    # 2. Prediction Results
    # 2. Prediction Results
    st.subheader("2. Prediction Results")
    
    # Ensure Predicted_Churn is integer for consistent plotting
    input_df['Predicted_Churn'] = input_df['Predicted_Churn'].astype(int)

    # Metrics
    churn_rate = input_df['Predicted_Churn'].mean()
    st.metric(label="Predicted Churn Rate", value=f"{churn_rate:.2%}")
    
    # Layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Churn Distribution")
        # Donut Chart
        churn_counts = input_df['Predicted_Churn'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'], wedgeprops=dict(width=0.3))
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1, use_container_width=True)
        
    with col2:
        st.markdown("### Churn Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.histplot(input_df['Churn_Probability'], bins=20, kde=True, ax=ax2, color='skyblue')
        ax2.set_xlabel("Churn Probability")
        ax2.set_ylabel("Count")
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Key Insights & Drivers")
    
    # Correlation Heatmap (Numerical)
    st.markdown("#### Correlation with Churn Probability")
    # Select numerical columns for correlation
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove predicted columns for correlation with features, but keep Churn_Probability to see what correlates with it
    corr_cols = [c for c in numeric_cols if c not in ['Predicted_Churn', 'RowNumber', 'CustomerId']]
    
    if len(corr_cols) > 1:
        corr = input_df[corr_cols].corr()[['Churn_Probability']].sort_values(by='Churn_Probability', ascending=False)
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3)
        st.pyplot(fig3, use_container_width=True)

    # Categorical Analysis
    st.markdown("#### Churn Rate by Category")
    cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    
    selected_cat = st.selectbox("Select Category to Analyze", [c for c in cat_cols if c in input_df.columns])
    
    if selected_cat:
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        # Calculate mean churn probability by category
        cat_churn = input_df.groupby(selected_cat)['Churn_Probability'].mean().reset_index()
        sns.barplot(x=selected_cat, y='Churn_Probability', data=cat_churn, ax=ax4, palette='viridis')
        ax4.set_ylabel("Avg Churn Prob")
        ax4.set_ylim(0, 1)
        st.pyplot(fig4, use_container_width=True)

    # Numerical Analysis (Box Plots)
    st.markdown("#### Distribution of Numerical Features by Churn Risk")
    num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    selected_num = st.selectbox("Select Numerical Feature", [c for c in num_features if c in input_df.columns])
    
    if selected_num:
        fig5, ax5 = plt.subplots(figsize=(8, 3))
        # Ensure Predicted_Churn is int for palette, but provide robust palette just in case
        robust_palette = {0: '#66b3ff', 1: '#ff9999', '0': '#66b3ff', '1': '#ff9999'}
        sns.boxplot(x='Predicted_Churn', y=selected_num, data=input_df, ax=ax5, palette=robust_palette, hue='Predicted_Churn', legend=False)
        ax5.set_xticklabels(['Retained', 'Churned'])
        st.pyplot(fig5, use_container_width=True)

    # High Risk Customers
    st.markdown("---")
    st.subheader("4. High Risk Customers")
    threshold = st.slider("Select Probability Threshold for High Risk", 0.0, 1.0, 0.7)
    high_risk_df = input_df[input_df['Churn_Probability'] > threshold].sort_values(by='Churn_Probability', ascending=False)
    st.write(f"Found {len(high_risk_df)} customers with churn probability > {threshold}")
    st.dataframe(high_risk_df)
    
    # Feature Importance (Global)
    st.markdown("---")
    st.subheader("5. Global Feature Importance (Model)")
    try:
        # Try to load pre-calculated importance
        fi_df = pd.read_csv('c:/Task_02/feature_importance.csv')
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax6, palette='magma')
        st.pyplot(fig6, use_container_width=True)
    except:
        st.info("Feature importance file not found.")

else:
    st.info("Awaiting CSV file upload. Please upload the 'Churn_Modelling.csv' or similar dataset.")
    
    # Option to use demo data
    if st.button("Use Demo Data (Test Predictions)"):
        try:
            demo_df = pd.read_csv('c:/Task_02/test_predictions.csv')
            # Rename column to match app logic
            if 'Predicted_Exited' in demo_df.columns:
                demo_df.rename(columns={'Predicted_Exited': 'Predicted_Churn'}, inplace=True)
            
            # Ensure Predicted_Churn is integer
            demo_df['Predicted_Churn'] = demo_df['Predicted_Churn'].astype(int)
            
            st.subheader("Demo Data Preview")
            st.dataframe(demo_df.head())
            
            # Metrics
            churn_rate = demo_df['Predicted_Churn'].mean()
            st.metric(label="Predicted Churn Rate", value=f"{churn_rate:.2%}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Churn Distribution")
                churn_counts = demo_df['Predicted_Churn'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                ax1.pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'], wedgeprops=dict(width=0.3))
                ax1.axis('equal')
                st.pyplot(fig1, use_container_width=True)
            
            with col2:
                st.markdown("### Churn Probability Distribution")
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                sns.histplot(demo_df['Churn_Probability'], bins=20, kde=True, ax=ax2, color='orange')
                st.pyplot(fig2, use_container_width=True)
            
            st.markdown("#### Correlation with Churn Probability")
            numeric_cols = demo_df.select_dtypes(include=[np.number]).columns.tolist()
            corr_cols = [c for c in numeric_cols if c not in ['Predicted_Churn', 'RowNumber', 'CustomerId']]
            if len(corr_cols) > 1:
                corr = demo_df[corr_cols].corr()[['Churn_Probability']].sort_values(by='Churn_Probability', ascending=False)
                fig3, ax3 = plt.subplots(figsize=(8, 3))
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3)
                st.pyplot(fig3, use_container_width=True)

            st.subheader("Feature Importance")
            fi_df = pd.read_csv('c:/Task_02/feature_importance.csv')
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax3, palette='magma')
            st.pyplot(fig3, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not load demo data: {e}")
