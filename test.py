import pandas as  pd

df = pd.read_csv('schwing_stetter_sales_2022_2024.csv')
print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'].head())