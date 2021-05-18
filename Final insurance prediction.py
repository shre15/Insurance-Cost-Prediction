


import numpy as np # linear algebra
import pandas as pd # data processing

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns



# reading the data

data = pd.read_csv(r'‪C:\Users\shrey\Downloads\insurance.csv')

# checking the shape
print(data.shape)


# checking the head of the dataset

data.head()


# describing the data

data.describe()

# checking if the dataset contains any NULL values

data.isnull().any()


sns.pairplot(data)

# lmplot between age and charges

sns.lmplot('age', 'charges', data = data)


# bubble plot to show relation bet age, charges and children

plt.rcParams['figure.figsize'] = (15, 8)
plt.scatter(x = data['age'], y = data['charges'], s = data['children']*100, alpha = 0.2, color = 'red')
plt.title('Bubble plot', fontsize = 30)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()




# plotting the correlation plot for the dataset

f, ax = plt.subplots(figsize = (10, 10))

corr = data.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), 
            cmap = sns.diverging_palette(50, 10, as_cmap = True), square = True, ax = ax)

# removing unnecassary columns from the dataset

data = data.drop('region', axis = 1)

print(data.shape)

data.columns


# label encoding for sex and smoker

# importing label encoder
from sklearn.preprocessing import LabelEncoder

# creating a label encoder
le = LabelEncoder()


# label encoding for sex
# 0 for females and 1 for males
data['sex'] = le.fit_transform(data['sex'])

# label encoding for smoker
# 0 for smokers and 1 for non smokers
data['smoker'] = le.fit_transform(data['smoker'])


# splitting the dependent and independent variable

x = data.iloc[:,:5]
y = data.iloc[:,5]

print(x.shape)
print(y.shape)


# splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# standard scaling

from sklearn.preprocessing import StandardScaler

# creating a standard scaler
sc = StandardScaler()

# feeding independents sets into the standard scaler
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Set data
from math import pi
df = pd.DataFrame({
'group': [i for i in range(0, 1338)],
'Age': data['age'],
'Charges': data['charges'],
'Children': data['children'],
'BMI': data['bmi']
})
 

# importing the model
# Multiple linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# Fit linear model by passing training dataset
model.fit(x_train,y_train)

# Predicting the target variable for test datset
predictions = model.predict(x_test)

print('THE PREDICTION VALUE IN MULTIPLE LINEAR REGRESSOR IS:\n')
print(predictions)

#plotting the y prediction
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.title('Multiple Linear Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# REGRESSION ANALYSIS
# RANDOM FOREST


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# creating the model
model = RandomForestRegressor(n_estimators = 40, max_depth = 4, n_jobs = -1)

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)
print('THE PREDICTION VALUE OF IN RANDOM FOREST REGRESSOR IS:\n')
print(y_pred)

#plotting the y prediction
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.title('Random Forest Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# feature extraction

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# importing the model
# Multiple linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# Fit linear model by passing training dataset
model.fit(x_train,y_train)

# Predicting the target variable for test datset
predictions = model.predict(x_test)

print('THE PREDICTION VALUE OF PCA WITH MULTIPLE LINEAR REGRESSOR IS:\n')
print(predictions)

#plotting the y prediction
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.title('PCA with Multiple Linear Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# feature extraction

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


# REGRESSION ANALYSIS
# RANDOM FOREST


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# creating the model
model = RandomForestRegressor(n_estimators = 40, max_depth = 4, n_jobs = -1)

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

print('THE PREDICTION VALUE OF IN PCA WITH RANDOM FOREST REGRESSOR IS:\n')
print(y_pred)

#plotting the y prediction
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.title('PCA with Random Forest Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)

# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

# Calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2 score :", r2)