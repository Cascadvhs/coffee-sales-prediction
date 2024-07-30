import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Загружаю датасет с информацией о продажах кофе. Провожу обязательные проверки EDA
df = pd.read_csv('/kaggle/input/coffee-sales/index.csv')
df.head(10)
df.columns
df.dtypes
df.info() # Общая информация о таблице
df.describe() # Идентефикация выбросов (выбивающиеся из логики значение (слишком большие и малые))
df.duplicated().sum() # Проверка дубликатов
df.isnull().sum() # Проверка на пустые значения 

# Список категориальных столбцов
categorical_columns = ['cash_type', 'coffee_name']

# Уникальные значения для каждого категориального столбца
for column in categorical_columns:
    unique_values = df[column].unique()
    print(f"Уникальные значения в '{column}': {unique_values}\n")
    
# Распределение потраченных денег
plt.figure(figsize=(10, 6))
sns.histplot(df['money'], bins=20, kde=True)
plt.title('Распределение потраченных денег')
plt.xlabel('Деньги $')
plt.ylabel('Частота')
plt.show()

# Транзакции по времени
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = pd.to_datetime(df['datetime']).dt.hour

plt.figure(figsize=(10, 6))
sns.countplot(x='month', data=df)
plt.title('Транзакций в месяц')
plt.xlabel('Месяц')
plt.ylabel('Число транзакций')
plt.show()

# Популярность разновидностей кофе
plt.figure(figsize=(12, 6))
sns.countplot(y='coffee_name', data=df, order=df['coffee_name'].value_counts().index)
plt.title('Популярность разновидностей кофе')
plt.xlabel('Количество транзакций')
plt.ylabel('Разновидность кофе')
plt.show()

# Убираю неиспользуемые столбцы
df.drop(columns = ['date', 'datetime', 'card'], inplace = True)
df.info()

# Преобразую тип наличных денег с помощью LabelEncoder
label_encoder = LabelEncoder()
df['cash_type'] = label_encoder.fit_transform(df['cash_type'])

# Преобразую coffee_name с помощью One Hot Encoder
df = pd.get_dummies(df, columns = ['coffee_name'], drop_first = True)

coffee_name_columns = [col for col in df.columns if 'coffee_name' in col]
df[coffee_name_columns] = df[coffee_name_columns].astype(int)

# Проверка, что используем наш Датафрейм
X = df.drop(columns=['money'])
y = df['money']

# Сплитуем датасет
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Список моделей
models = [
    LinearRegression(),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    xgb.XGBRegressor(eval_metric='rmse', random_state=42),
]

# Название моделей
model_names = [
    "Линейная регрессия",
    "Дерево решений",
    "Метод случайного леса",
    "Gradient Boosting",
    "XGBoost",
]

# Обучение моделей
results = {}

for name, model in zip(model_names, models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R²': r2}

# Отображение результатов
for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['MSE']}, R²: {metrics['R²']}")