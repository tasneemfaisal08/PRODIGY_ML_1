import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv(r"C:\Users\reBuyTech\Desktop\prodigy_intern\task1\train.csv")
test_data=pd.read_csv(r"C:\Users\reBuyTech\Desktop\prodigy_intern\task1\test.csv")
sample_submission=pd.read_csv(r"C:\Users\reBuyTech\Desktop\prodigy_intern\task1\sample_submission.csv")

print("Train:", train_data.shape)
print("Test:",test_data.shape)
print("Sample:", sample_submission.shape)

train_data.head()

test_data.head()

sample_submission.head()

print(train_data.columns)

train_data.info()

train_data.isnull().sum()

null_columns = [(col, train_data[col].isnull().sum()) for col in train_data.columns if train_data[col].isnull().sum() > 0]

print("Columns with null values in the training data:")
for col_name, null_count in null_columns:
    print(f"{col_name} - {null_count} null values")


import numpy as np
# Calculate mean for numerical columns to fill null values
mean_values_train = train_data.mean(numeric_only=True)

fill_values_train = {}

for col_name in train_data.columns:
    dtype = train_data[col_name].dtype
    if dtype in [np.float64, np.float32]:
        fill_values_train[col_name] = mean_values_train[col_name]  
    elif dtype in [np.int64, np.int32]:
        fill_values_train[col_name] = mean_values_train[col_name]  
    elif dtype == object:
        fill_values_train[col_name] = train_data[col_name].mode()[0]  

# Fill null values in the training data
train_data_filled = train_data.fillna(fill_values_train)

train_data_filled.head()

train_data.describe()

corr = train_data.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()
sns.set(font_scale=1)  
sns.heatmap(corr, vmax=1, square=True)

print(test_data.columns)

test_data.info()

test_data.isnull().sum()

null_columns = [(col, test_data[col].isnull().sum()) for col in test_data.columns if test_data[col].isnull().sum() > 0]

print("Columns with null values in the testing data:")
for col_name, null_count in null_columns:
    print(f"{col_name} - {null_count} null values")

import numpy as np
# Calculate mean for numerical columns to fill null values
mean_values_test = test_data.mean(numeric_only=True)

fill_values_test = {}

for col_name in test_data.columns:
    dtype = test_data[col_name].dtype
    if dtype in [np.float64, np.float32]:
        fill_values_test[col_name] = mean_values_test[col_name]  
    elif dtype in [np.int64, np.int32]:
        fill_values_test[col_name] = mean_values_test[col_name]  
    elif dtype == object:
        fill_values_test[col_name] = test_data[col_name].mode()[0]    

# Fill null values in the testing data
test_data_filled = test_data.fillna(fill_values_test)

test_data_filled.head()

X_train = train_data_filled[['MasVnrArea', 'BedroomAbvGr', 'FullBath']]
y_train= train_data_filled['SalePrice'] 

X_test=test_data_filled [['MasVnrArea', 'BedroomAbvGr', 'FullBath']]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

data = {'prediction': y_pred,
        'Actual':sample_submission['SalePrice'] }
df = pd.DataFrame(data)
print(df)

mse = mean_squared_error(sample_submission['SalePrice'], y_pred)
print(f"\nMean Squared Error: {mse}")

# Visualize the results
plt.figure(figsize=(14, 6))

# Scatter plot: Actual vs Predicted prices
if sample_submission['SalePrice'] is not None:
    plt.subplot(1, 2, 1)
    plt.scatter(sample_submission['SalePrice'], y_pred, alpha=0.5)
    plt.plot([sample_submission['SalePrice'].min(), sample_submission['SalePrice'].max()], [sample_submission['SalePrice'].min(), sample_submission['SalePrice'].max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')

    # Residuals plot
    residuals = sample_submission['SalePrice']- y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='dashed')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')

plt.tight_layout()
plt.show()



