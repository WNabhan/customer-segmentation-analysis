import pandas as pd

#Lendo todos os dados
aisle = pd.read_csv('data/raw/aisles.csv')
departments = pd.read_csv('data/raw/departments.csv')
products = pd.read_csv('data/raw/products.csv')
orders = pd.read_parquet('data/raw/orders.parquet')
order_products = pd.read_parquet('data/raw/order_products.parquet')

#Consolidando todos os dados em um só dataframe com base nos ids
data = pd.merge(order_products, products, on='product_id', how = 'left')
data = pd.merge(data, aisle, on='aisle_id', how = 'left')
data = pd.merge(data, departments, on='department_id', how = 'left')
data = pd.merge(data, orders, on='order_id', how = 'left')

#Removendo as colunas de ids e reordenando as colunas do dataframe final
data.drop(columns = ['product_id', 'aisle_id', 'department_id'] , axis=1, inplace=True)
data = data[['user_id', 'order_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'product_name', 'aisle', 'department', 'add_to_cart_order', 'reordered']]

#Ordenando o dataframe por user_id e order_number na sequencia
data = data.sort_values(by=['user_id', 'order_number'], ascending=[True, True])

#Salvando em parquet
data.to_parquet('data/processed/data.parquet', index=False)