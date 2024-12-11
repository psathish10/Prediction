# import pandas as pd

# # Creating a sample dataset for sales prediction
# data = {
#     "Date": pd.date_range(start="2023-01-01", end="2024-01-01", freq="W"),
#     "Product_ID": ["BP350", "BP1400", "BP1800"] * 52,
#     "Region": ["North", "South", "East", "West"] * 39,
#     "Units_Sold": [int(x) for x in pd.np.random.randint(10, 100, size=156)],
#     "Price_Per_Unit": [int(x) for x in pd.np.random.randint(1000, 5000, size=156)],
#     "Delivery_Time": [int(x) for x in pd.np.random.randint(2, 15, size=156)],
#     "Seasonality": ["High", "Medium", "Low"] * 52
# }

# # Creating the DataFrame
# sales_data = pd.DataFrame(data)

# # Save the dataset as a CSV file for user download
# file_path = "/mnt/data/sales_data.csv"
# sales_data.to_csv(file_path, index=False)
# file_path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sales_data(n_rows=2000):
    # Define base data structures
    countries = ['Germany', 'Austria', 'USA', 'Brazil', 'Russia', 'UK', 'India', 'France', 
                'Netherlands', 'Sweden', 'China', 'Australia']
    
    states = {
        'India': ['Tamil Nadu', 'Maharashtra', 'Delhi', 'Karnataka', 'West Bengal', 
                 'Gujarat', 'Kerala', 'Punjab', 'Telangana'],
        'USA': ['California', 'Texas', 'New York', 'Florida', 'Illinois'],
        'Germany': ['Bavaria', 'Baden-Württemberg', 'North Rhine-Westphalia', 'Hesse', 'Berlin'],
        'China': ['Beijing', 'Shanghai', 'Guangdong', 'Sichuan', 'Jiangsu'],
        'DEFAULT': ['State 1', 'State 2', 'State 3']
    }
    
    product_categories = {
        'Concrete Preparation': [
            ('Concrete Batching Plant', 150000, 300000),
            ('Transit Mixer', 75000, 120000)
        ],
        'Concrete Pumps': [
            ('Stationary Pump BP350', 80000, 120000),
            ('Stationary Pump BP1400', 120000, 180000),
            ('Stationary Pump BP1800', 150000, 220000),
            ('Mobile Pump', 200000, 350000)
        ],
        'Placing Equipment': [
            ('Concrete Placing Boom', 90000, 150000),
            ('Separate Placing Boom', 70000, 120000)
        ],
        'Recycling Solutions': [
            ('Recycling Plant', 180000, 300000),
            ('Washing Plant', 120000, 200000)
        ],
        'Special Equipment': [
            ('Sludge Pump', 40000, 80000),
            ('Trenchless Equipment', 100000, 180000),
            ('Foundation Equipment', 150000, 250000)
        ]
    }

    # Generate evenly distributed dates for 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_rows)
    
    # Generate sales records
    records = []
    for date in dates:
        # Add some seasonality - higher sales in construction season
        if date.month in [3, 4, 5, 9, 10, 11]:  # Peak construction months
            country_weights = [0.15, 0.1, 0.15, 0.1, 0.1, 0.05, 0.2, 0.05, 0.03, 0.02, 0.03, 0.02]
        else:
            country_weights = [0.1, 0.08, 0.1, 0.08, 0.08, 0.08, 0.15, 0.08, 0.08, 0.07, 0.05, 0.05]
            
        country = random.choices(countries, weights=country_weights)[0]
        state = random.choice(states.get(country, states['DEFAULT']))
        category = random.choice(list(product_categories.keys()))
        product, min_price, max_price = random.choice(product_categories[category])
        
        # Generate realistic sales data with some seasonal variation
        season_multiplier = 1.1 if date.month in [3, 4, 5, 9, 10, 11] else 0.9
        base_price = random.uniform(min_price, max_price) * season_multiplier
        quantity = random.randint(1, 3)
        discount_percent = random.uniform(0, 0.15)
        
        record = {
            'Date': date,
            'Country': country,
            'State': state,
            'Product_Category': category,
            'Product_Name': product,
            'Quantity': quantity,
            'Base_Price_EUR': round(base_price, 2),
            'Discount_Percentage': round(discount_percent * 100, 2),
            'Final_Price_EUR': round(base_price * (1 - discount_percent), 2),
            'Total_Sale_EUR': round(base_price * (1 - discount_percent) * quantity, 2),
            'Order_ID': f"ORD-{date.strftime('%Y%m')}-{random.randint(1000, 9999)}",
            'Dealer_ID': f"DEL-{random.randint(100, 999)}",
            'Warranty_Years': random.choice([1, 2, 3, 5]),
            'Payment_Terms': random.choice(['30 Days', '60 Days', '90 Days', 'Immediate']),
            'Delivery_Terms': random.choice(['FOB', 'CIF', 'EXW', 'DDP'])
        }
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    return df

# Generate sample data
sales_df = generate_sales_data(2000)

# Save to CSV
sales_df.to_csv('schwing_stetter_sales.csv', index=False)

# Display sample info
print("\nData Structure:")
print(sales_df.info())
print("\nSample of first few records:")
print(sales_df.head())
print("\nSummary statistics:")
print(sales_df.describe())