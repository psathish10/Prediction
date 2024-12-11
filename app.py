import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from PIL import Image
import google.generativeai as genai

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCJVYkpDOt6-hdf8CTTDI6nPUkJE7y4Rs0")

# Helper Functions
def advanced_feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    data['Year'] = data['Date'].dt.year
    data['Quarter'] = data['Date'].dt.quarter
    data['Week_of_Year'] = data['Date'].dt.isocalendar().week
    data['Revenue_Rolling_Mean_3M'] = data.groupby(['Region', 'Product_ID'])['Revenue'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    data['Revenue_Rolling_Std_3M'] = data.groupby(['Region', 'Product_ID'])['Revenue'].transform(lambda x: x.rolling(window=3, min_periods=1).std())
    data['Revenue_Lag_1'] = data.groupby(['Region', 'Product_ID'])['Revenue'].shift(1)
    return data.dropna()

def generate_ai_summary(title, description, key_insights):
    """
    Generate AI-powered summaries for each chart.
    """
    prompt = f"""
    Create a professional summary for a CFO and CSO based on the following:
    
    Title: {title}
    Description: {description}
    Key Insights: {key_insights}

    Ensure the summary provides strategic recommendations.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Summary generation failed: {e}"

def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def main():
    st.set_page_config(layout="wide", page_title="CFO & CSO Sales Dashboard")

    # Sidebar with company logo
    st.sidebar.image("logo.png", use_container_width=True)
    st.sidebar.title("Strategic Sales & Revenue Dashboard")

    st.title("🚀 **CFO & CSO Sales Performance Dashboard**")

    # File Upload
    uploaded_file = st.file_uploader("Upload Sales Data (CSV format)", type="csv")
    if uploaded_file:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Feature Engineering
        data = advanced_feature_engineering(data)

        # Prepare Features and Target
        features = ['Month', 'Day_of_Week', 'Year', 'Quarter', 'Week_of_Year', 'Revenue_Rolling_Mean_3M', 'Revenue_Rolling_Std_3M', 'Revenue_Lag_1']
        X = data[features]
        y = data['Revenue']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Models
        models = train_models(X_train, y_train)

        # AI Summary Placeholder
        ai_summaries = {}

        # Revenue Insights
        st.header("💰 **Revenue Insights**")
        monthly_revenue = data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
        fig_revenue = px.line(monthly_revenue, x="Month", y="Revenue", color="Year", markers=True, title="Monthly Revenue Trends")
        st.plotly_chart(fig_revenue)
        ai_summaries['Revenue Insights'] = generate_ai_summary(
            "Monthly Revenue Trends",
            "A line chart depicting revenue trends across different months for each year.",
            f"Total Revenue: ${data['Revenue'].sum():,.2f}, Average Monthly Revenue: ${data['Revenue'].mean():,.2f}"
        )
        st.info(ai_summaries['Revenue Insights'])

        # Product-Wise Revenue Distribution
        st.header("📦 **Product-Wise Revenue Distribution**")
        fig_box = px.box(data, x="Product_ID", y="Revenue", color="Region", title="Revenue Distribution by Product and Region")
        st.plotly_chart(fig_box)
        ai_summaries['Product Revenue'] = generate_ai_summary(
            "Product Revenue Distribution",
            "Box plot showing revenue distribution across products by region.",
            "Key outliers indicate potential for high-performing products."
            f"Data  suggests that X axis Product ID ${data['Product_ID']}, Y Axis is Revenue  ${data['Revenue']}, color used by Region are  ${data['Region']}, products."
        )
        st.info(ai_summaries['Product Revenue'])

        # Region-Wise Revenue Contribution
        st.header("🌍 **Region-Wise Revenue Contribution**")
        region_revenue = data.groupby('Region')['Revenue'].sum().reset_index()
        fig_region = px.pie(region_revenue, values='Revenue', names='Region', title="Region-Wise Revenue Contribution")
        st.plotly_chart(fig_region)
        ai_summaries['Region Revenue'] = generate_ai_summary(
            "Region-Wise Revenue Contribution",
            "Pie chart showing the share of revenue contributed by each region.",
            "Highlight regions with strong performance and opportunities for growth."
            f"Data  suggests that Values Revenue ${data['Revenue']},name is Region  ${data['Region']}."
        )
        st.info(ai_summaries['Region Revenue'])

        # Model Forecasts
        st.header("🔮 **Forecasting Results**")
        for model_name, model in models.items():
            forecast = model.predict(X_test[:1])  # Example forecast
            st.metric(f"{model_name} Forecast", f"${forecast[0]:,.2f}")

        st.header("📈 **Feature Importance** (Random Forest)")
        feature_importance = models["Random Forest"].feature_importances_
        feature_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        fig_features = px.bar(feature_df, x="Importance", y="Feature", orientation="h", title="Feature Importance for Revenue Prediction")
        st.plotly_chart(fig_features)

if __name__ == "__main__":
    main()
