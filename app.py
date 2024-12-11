import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCJVYkpDOt6-hdf8CTTDI6nPUkJE7y4Rs0")

# Helper Functions
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
    # st.image("logo.png", use_container_width=True)
    st.title("Strategic Sales & Revenue Dashboard")

    st.title("🚀 **CFO & CSO Sales Performance Dashboard**")


    data = pd.read_csv("schwing_stetter_sales.csv")

        # Data Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Quarter'] = data['Date'].dt.quarter
    data['Week_of_Year'] = data['Date'].dt.isocalendar().week

        # Feature Preparation
    features = ['Month', 'Year', 'Quantity', 'Base_Price_EUR', 'Discount_Percentage']
    X = data[features]
    y = data['Total_Sale_EUR']

        # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Models
    models = train_models(X_train, y_train)

    ai_summaries = {}

        # 1. Total Sales by Product Category
    st.header("📊 **Total Sales by Product Category**")
    category_sales = data.groupby('Product_Category')['Total_Sale_EUR'].sum().reset_index()
    fig_category = px.bar(category_sales, x="Product_Category", y="Total_Sale_EUR", title="Total Sales by Product Category")
    st.plotly_chart(fig_category)
    ai_summaries['Category Sales'] = generate_ai_summary(
            "Total Sales by Product Category",
            "A bar chart showing the total sales by each product category.",
            f"Total Sales: ${category_sales['Total_Sale_EUR'].sum():,.2f}, Average Sales per Category: ${category_sales['Total_Sale_EUR'].mean():,.2f}"
        )
    st.info(ai_summaries['Category Sales'])

        # 2. Sales by Country
    st.header("🌍 **Sales by Country**")
    country_sales = data.groupby('Country')['Total_Sale_EUR'].sum().reset_index()
    fig_country = px.pie(country_sales, values='Total_Sale_EUR', names='Country', title="Sales Distribution by Country")
    st.plotly_chart(fig_country)
    ai_summaries['Country Sales'] = generate_ai_summary(
            "Sales by Country",
            "A pie chart showing the sales distribution across different countries.",
            f"Total Sales: ${country_sales['Total_Sale_EUR'].sum():,.2f}. The country with the highest sales is {country_sales.loc[country_sales['Total_Sale_EUR'].idxmax(), 'Country']}."
        )
    st.info(ai_summaries['Country Sales'])

        # 3. Sales vs Quantity
    st.header("📈 **Sales vs Quantity Sold**")
    fig_sales_quantity = px.scatter(data, x="Quantity", y="Total_Sale_EUR", title="Sales vs Quantity Sold")
    st.plotly_chart(fig_sales_quantity)
    ai_summaries['Sales vs Quantity'] = generate_ai_summary(
            "Sales vs Quantity Sold",
            "A scatter plot showing the relationship between the quantity sold and the total sales.",
            "A positive correlation between quantity sold and total sales can be observed, indicating that higher quantities tend to increase sales."
        )
    st.info(ai_summaries['Sales vs Quantity'])

        # 4. Sales Over Time (Monthly Trends)
    st.header("📅 **Sales Over Time (Monthly Trends)**")
    monthly_sales = data.groupby(['Year', 'Month'])['Total_Sale_EUR'].sum().reset_index()
    fig_sales_time = px.line(monthly_sales, x="Month", y="Total_Sale_EUR", color="Year", markers=True, title="Monthly Sales Trends")
    st.plotly_chart(fig_sales_time)
    ai_summaries['Monthly Sales'] = generate_ai_summary(
            "Sales Over Time",
            "A line chart showing the total sales over time (monthly).",
            f"Total Sales in the last year: ${monthly_sales['Total_Sale_EUR'].sum():,.2f}, Average Monthly Sales: ${monthly_sales['Total_Sale_EUR'].mean():,.2f}"
        )
    st.info(ai_summaries['Monthly Sales'])

        # 5. Sales by Payment Terms
    st.header("💳 **Sales by Payment Terms**")
    payment_sales = data.groupby('Payment_Terms')['Total_Sale_EUR'].sum().reset_index()
    fig_payment = px.pie(payment_sales, values='Total_Sale_EUR', names='Payment_Terms', title="Sales Distribution by Payment Terms")
    st.plotly_chart(fig_payment)
    ai_summaries['Payment Terms Sales'] = generate_ai_summary(
            "Sales by Payment Terms",
            "A pie chart showing the sales distribution across different payment terms.",
            f"Major payment terms include: {', '.join(payment_sales['Payment_Terms'].unique())}. The highest share of sales is under {payment_sales.loc[payment_sales['Total_Sale_EUR'].idxmax(), 'Payment_Terms']}."
        )
    st.info(ai_summaries['Payment Terms Sales'])

        # Model Forecasts
    st.header("🔮 **Forecasting Results**")
    for model_name, model in models.items():
            forecast = model.predict(X_test[:1])  # Example forecast
            st.metric(f"{model_name} Forecast", f"${forecast[0]:,.2f}")

        # Feature Importance (Random Forest)
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
