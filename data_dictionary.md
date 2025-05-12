## Dicionário de Dados

### `aisles.csv`

| Campo     | Tipo      | Descrição                 |
|-----------|-----------|---------------------------|
| aisle_id  | Numérico  | Identificador da seção    |
| aisle     | Categórico| Nome da seção             |

### `departments.csv`

| Campo         | Tipo      | Descrição                    |
|---------------|-----------|------------------------------|
| department_id | Numérico  | Identificador do departamento|
| department    | Categórico| Nome do departamento         |

### `products.csv`

| Campo         | Tipo      | Descrição                        |
|---------------|-----------|----------------------------------|
| product_id    | Numérico  | Identificador do produto         |
| product_name  | Categórico| Nome do produto                  |
| aisle_id      | Numérico  | Identificador da seção           |
| department_id | Numérico  | Identificador do departamento    |

### `orders.parquet`

| Campo                   | Tipo      | Descrição                                                                 |
|-------------------------|-----------|---------------------------------------------------------------------------|
| order_id                | Numérico  | Identificador do pedido                                                   |
| user_id                 | Numérico  | Identificador do usuário                                                  |
| eval_set                | Categórico| Indica se o pedido pertence ao conjunto ‘prior’, ‘train’ ou ‘test’       |
| order_number            | Numérico  | Número do pedido do usuário                                               |
| order_dow               | Numérico  | Dia da semana do pedido (0 = domingo, 6 = sábado)                         |
| order_hour_of_day       | Numérico  | Hora do pedido                                                            |
| days_since_prior_order  | Numérico  | Intervalo de dias desde o pedido anterior                                 |

### `order_products.parquet`

| Campo              | Tipo      | Descrição                                                                     |
|--------------------|-----------|---------------------------------------------------------------------------------|
| order_id           | Numérico  | Identificador do pedido                                                       |
| product_id         | Numérico  | Identificador do produto                                                      |
| add_to_cart_order  | Numérico  | Ordem de adição do produto ao carrinho                                        |
| reordered          | Numérico  | Indica se o produto é uma recompra em relação ao histórico do usuário (0 = primeira compra, 1 = item já adquirido anteriormente)           |