import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime
from datetime import timedelta
from app import main

# Set page config
st.set_page_config(layout="wide", page_title="Schwing Stetter Sales Analytics")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('schwing_stetter_sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.image('logo.png', width=250)
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select Page', ['Sales Analytics', 'Sales Forecasting','Ai Dashboard'])

# Sales Analytics Page
if page == 'Sales Analytics':
    st.title('Schwing Stetter Sales Analytics Dashboard')
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${df['Total_Sale_EUR'].sum():,.0f}")
    with col2:
        st.metric("Total Orders", len(df))
    with col3:
        st.metric("Average Order Value", f"${df['Total_Sale_EUR'].mean():,.0f}")
    with col4:
        st.metric("Total Units Sold", df['Quantity'].sum())

    # Sales by Country Map
    st.subheader("Global Sales Distribution")
    sales_by_country = df.groupby('Country')['Total_Sale_EUR'].sum().reset_index()
    fig_map = px.choropleth(sales_by_country, 
                           locations='Country', 
                           locationmode='country names',
                           color='Total_Sale_EUR',
                           hover_name='Country',
                           color_continuous_scale='Viridis')
    st.plotly_chart(fig_map, use_container_width=True)

    # Sales Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Sales Trend")
        monthly_sales = df.groupby(['Year', 'Month'])['Total_Sale_EUR'].sum().reset_index()
        fig_trend = px.line(monthly_sales, 
                           x='Month', 
                           y='Total_Sale_EUR',
                           color='Year',
                           title='Monthly Sales Trend by Year')
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("Product Category Distribution")
        fig_donut = px.pie(df, 
                          values='Total_Sale_EUR', 
                          names='Product_Category',
                          hole=0.4,
                          title='Sales by Product Category')
        st.plotly_chart(fig_donut, use_container_width=True)

    # Product Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Products by Revenue")
        top_products = df.groupby('Product_Name')['Total_Sale_EUR'].sum().sort_values(ascending=True).tail(10)
        fig_products = px.bar(top_products, 
                            orientation='h',
                            title='Top 10 Products by Revenue')
        st.plotly_chart(fig_products, use_container_width=True)

    with col2:
        st.subheader("Quantity vs Price Distribution")
        fig_scatter = px.scatter(df, 
                               x='Base_Price_EUR', 
                               y='Quantity',
                               color='Product_Category',
                               size='Total_Sale_EUR',
                               title='Price vs Quantity Distribution')
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Payment and Delivery Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Payment Terms Distribution")
        fig_payment = px.bar(df['Payment_Terms'].value_counts(),
                           title='Distribution of Payment Terms')
        st.plotly_chart(fig_payment, use_container_width=True)

    with col2:
        st.subheader("Warranty Years Distribution")
        fig_warranty = px.histogram(df, 
                                  x='Warranty_Years',
                                  title='Distribution of Warranty Years')
        st.plotly_chart(fig_warranty, use_container_width=True)
elif page == 'Ai Dashboard':
   main()

# Sales Forecasting Page
elif page == 'Sales Forecasting':
    st.title('Advanced Sales Forecasting')
    
    # Create tabs for different forecasting views
    forecast_tabs = st.tabs(['Overall Forecast', 'State-wise Forecast', 'Product-wise Forecast'])
    
    # Overall Forecast Tab
    with forecast_tabs[0]:
        st.subheader('Overall Sales Forecast')
        
        @st.cache_data
        def prepare_forecast_data(df):
            # Create features
            df_model = df.copy()
            
            # Encode categorical variables
            le = LabelEncoder()
            df_model['Country_encoded'] = le.fit_transform(df_model['Country'])
            df_model['Product_Category_encoded'] = le.fit_transform(df_model['Product_Category'])
            df_model['Product_Name_encoded'] = le.fit_transform(df_model['Product_Name'])
            
            # Create time-based features
            df_model['Year_num'] = df_model['Year'] - df_model['Year'].min()
            
            # Select features for modeling
            features = ['Year_num', 'Month', 'Quarter', 'Country_encoded', 
                       'Product_Category_encoded', 'Product_Name_encoded', 'Quantity']
            
            X = df_model[features]
            y = df_model['Total_Sale_EUR']
            
            return X, y, features

        X, y, features = prepare_forecast_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        model_choice = st.selectbox('Select Model', 
                                   ['Decision Tree', 'Random Forest', 'XGBoost', 'Linear Regression'])

        @st.cache_resource
        def train_model(model_name, X_train, X_test, y_train, y_test):
            if model_name == 'Decision Tree':
                model = DecisionTreeRegressor(random_state=42)
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(random_state=42)
            elif model_name == 'XGBoost':
                model = XGBRegressor(random_state=42)
            else:
                model = LinearRegression()
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model, mse, r2, y_pred

        model, mse, r2, y_pred = train_model(model_choice, X_train, X_test, y_train, y_test)

        # Display model performance
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:,.2f}")
        with col2:
            st.metric("R² Score", f"{r2:.3f}")

        # Generate forecast
        forecast_months = st.slider("Select forecast period (months)", 1, 12, 6)
        
        if st.button("Generate Overall Forecast"):
            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=forecast_months+1, freq='M')[1:]
            
            future_df = pd.DataFrame()
            future_df['Date'] = future_dates
            future_df['Year'] = future_df['Date'].dt.year
            future_df['Month'] = future_df['Date'].dt.month
            future_df['Quarter'] = future_df['Date'].dt.quarter
            future_df['Year_num'] = future_df['Year'] - df['Year'].min()
            
            for col in ['Country_encoded', 'Product_Category_encoded', 'Product_Name_encoded', 'Quantity']:
                future_df[col] = X[col].mean()
            
            predictions = model.predict(future_df[features])
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['Total_Sale_EUR'],
                name='Historical Sales',
                line=dict(color='blue')
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name='Forecasted Sales',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_forecast, use_container_width=True)

    # State-wise Forecast Tab
    with forecast_tabs[1]:
        st.subheader('State-wise Sales Forecast')
        
        selected_country = st.selectbox('Select Country', df['Country'].unique())
        states_in_country = df[df['Country'] == selected_country]['State'].unique()
        selected_state = st.selectbox('Select State', states_in_country)
        
        state_df = df[(df['Country'] == selected_country) & (df['State'] == selected_state)].copy()
        
        @st.cache_data
        def prepare_state_forecast_data(state_df):
            df_model = state_df.copy()
            
            df_model['Month_sin'] = np.sin(2 * np.pi * df_model['Month']/12)
            df_model['Month_cos'] = np.cos(2 * np.pi * df_model['Month']/12)
            df_model['Year_num'] = df_model['Year'] - df_model['Year'].min()
            
            le = LabelEncoder()
            df_model['Product_Category_encoded'] = le.fit_transform(df_model['Product_Category'])
            df_model['Product_Name_encoded'] = le.fit_transform(df_model['Product_Name'])
            
            features = ['Year_num', 'Month_sin', 'Month_cos', 'Quarter',
                       'Product_Category_encoded', 'Product_Name_encoded', 'Quantity']
            
            X = df_model[features]
            y = df_model['Total_Sale_EUR']
            
            return X, y, features

        X_state, y_state, state_features = prepare_state_forecast_data(state_df)
        X_train_state, X_test_state, y_train_state, y_test_state = train_test_split(
            X_state, y_state, test_size=0.2, random_state=42)
        
        state_model = RandomForestRegressor(random_state=42)
        state_model.fit(X_train_state, y_train_state)
        
        forecast_months = st.slider("Select forecast period (months)", 1, 12, 6, key='state_forecast')
        
        if st.button("Generate State Forecast"):
            last_date = state_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=forecast_months+1, freq='M')[1:]
            
            future_state_df = pd.DataFrame()
            future_state_df['Date'] = future_dates
            future_state_df['Year'] = future_state_df['Date'].dt.year
            future_state_df['Month'] = future_state_df['Date'].dt.month
            future_state_df['Quarter'] = future_state_df['Date'].dt.quarter
            future_state_df['Year_num'] = future_state_df['Year'] - state_df['Year'].min()
            future_state_df['Month_sin'] = np.sin(2 * np.pi * future_state_df['Month']/12)
            future_state_df['Month_cos'] = np.cos(2 * np.pi * future_state_df['Month']/12)
            
            for col in ['Product_Category_encoded', 'Product_Name_encoded', 'Quantity']:
                future_state_df[col] = X_state[col].mean()
            
            state_predictions = state_model.predict(future_state_df[state_features])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_state_forecast = go.Figure()
                fig_state_forecast.add_trace(go.Scatter(
                    x=state_df['Date'],
                    y=state_df['Total_Sale_EUR'],
                    name='Historical Sales',
                    line=dict(color='blue')
                ))
                fig_state_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=state_predictions,
                    name='Forecasted Sales',
                    line=dict(color='red', dash='dash')
                ))
                fig_state_forecast.update_layout(title=f'Sales Forecast for {selected_state}')
                st.plotly_chart(fig_state_forecast)
            
            with col2:
                monthly_growth = pd.DataFrame({
                    'Date': future_dates,
                    'Forecasted_Sales': state_predictions
                })
                monthly_growth['Growth_Rate'] = monthly_growth['Forecasted_Sales'].pct_change() * 100
                
                fig_growth = px.bar(monthly_growth, x='Date', y='Growth_Rate',
                                  title='Forecasted Monthly Growth Rate (%)')
                st.plotly_chart(fig_growth)

    # Product-wise Forecast Tab
    with forecast_tabs[2]:
        st.subheader('Product-wise Sales Forecast')
        
        selected_category = st.selectbox('Select Product Category', df['Product_Category'].unique())
        products_in_category = df[df['Product_Category'] == selected_category]['Product_Name'].unique()
        selected_product = st.selectbox('Select Product', products_in_category)
        
        product_df = df[df['Product_Name'] == selected_product].copy()
        
        @st.cache_data
        def prepare_product_forecast_data(product_df):
            df_model = product_df.copy()
            
            df_model['Month_sin'] = np.sin(2 * np.pi * df_model['Month']/12)
            df_model['Month_cos'] = np.cos(2 * np.pi * df_model['Month']/12)
            df_model['Year_num'] = df_model['Year'] - df_model['Year'].min()
            
            le = LabelEncoder()
            df_model['Country_encoded'] = le.fit_transform(df_model['Country'])
            df_model['State_encoded'] = le.fit_transform(df_model['State'])
            
            features = ['Year_num', 'Month_sin', 'Month_cos', 'Quarter',
                       'Country_encoded', 'State_encoded', 'Quantity']
            
            X = df_model[features]
            y = df_model['Total_Sale_EUR']
            
            return X, y, features
        
        X_product, y_product, product_features = prepare_product_forecast_data(product_df)
        
        # Train product-specific model
        X_train_product, X_test_product, y_train_product, y_test_product = train_test_split(
            X_product, y_product, test_size=0.2, random_state=42)
        
        product_model = XGBRegressor(random_state=42)
        product_model.fit(X_train_product, y_train_product)
        
        # Product-specific forecasting
        forecast_months = st.slider("Select forecast period (months)", 1, 12, 6, key='product_forecast')
        
        if st.button("Generate Product Forecast"):
            # Generate future dates for product
            last_date = product_df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=forecast_months+1, freq='M')[1:]
            
            # Create forecast DataFrame
            future_product_df = pd.DataFrame()
            future_product_df['Date'] = future_dates
            future_product_df['Year'] = future_product_df['Date'].dt.year
            future_product_df['Month'] = future_product_df['Date'].dt.month
            future_product_df['Quarter'] = future_product_df['Date'].dt.quarter
            future_product_df['Year_num'] = future_product_df['Year'] - product_df['Year'].min()
            future_product_df['Month_sin'] = np.sin(2 * np.pi * future_product_df['Month']/12)
            future_product_df['Month_cos'] = np.cos(2 * np.pi * future_product_df['Month']/12)
            
            # Add other required features
            for col in ['Country_encoded', 'State_encoded', 'Quantity']:
                future_product_df[col] = X_product[col].mean()
            
            # Make product-specific predictions
            product_predictions = product_model.predict(future_product_df[product_features])
            
            # Create visualizations for product forecasts
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical vs Predicted Line Chart
                fig_product_forecast = go.Figure()
                fig_product_forecast.add_trace(go.Scatter(
                    x=product_df['Date'],
                    y=product_df['Total_Sale_EUR'],
                    name='Historical Sales',
                    line=dict(color='blue')
                ))
                fig_product_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=product_predictions,
                    name='Forecasted Sales',
                    line=dict(color='red', dash='dash')
                ))
                fig_product_forecast.update_layout(title=f'Sales Forecast for {selected_product}')
                st.plotly_chart(fig_product_forecast)
            
            with col2:
                # Price Trend Analysis
                fig_price = px.line(product_df, x='Date', y='Base_Price_EUR',
                                  title='Historical Price Trend')
                st.plotly_chart(fig_price)
            
            # Additional Product Analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Units Sold Forecast
                units_fig = px.bar(product_df.groupby('Month')['Quantity'].mean().reset_index(),
                                 x='Month', y='Quantity',
                                 title='Average Monthly Units Sold')
                st.plotly_chart(units_fig)
            
            with col2:
                # Discount Analysis
                discount_fig = px.scatter(product_df, x='Discount_Percentage', y='Total_Sale_EUR',
                                        title='Impact of Discounts on Sales')
                st.plotly_chart(discount_fig)

            # Add additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_monthly_sales = product_df['Total_Sale_EUR'].mean()
                st.metric("Average Monthly Sales", f"${avg_monthly_sales:,.2f}")
            with col2:
                total_quantity = product_df['Quantity'].sum()
                st.metric("Total Units Sold", f"{total_quantity:,}")
            with col3:
                avg_price = product_df['Base_Price_EUR'].mean()
                st.metric("Average Price", f"${avg_price:,.2f}")

# Add Export Functionality
if st.sidebar.button("Export Forecast Data"):
    # Create a DataFrame with forecasted values
    if 'predictions' in locals():
        export_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Sales': predictions
        })
        
        # Convert DataFrame to CSV
        csv = export_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

# Add footer
st.markdown("---")
st.markdown("Schwing Stetter Sales Analytics Dashboard - Powerd By Toolfe")

# Add error handling
def handle_errors():
    try:
        if not df.empty:
            pass
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

handle_errors()