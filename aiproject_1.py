import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sklearn
from sklearn.impute import SimpleImputer




df=pd.read_csv(r"C:/Users/GAMING STORE/Downloads/creative_datashit/pandas-master/50_startups.csv")



X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder           # Encoding categorical data
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
X = X[:, 1:]        # Avoiding the Dummy Variable Trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)
acc_score = regressor.score(X_test,y_test)
print(acc_score*100,'%')
y_pred = regressor.predict(X_test)      # Predicting the Test set results


print(regressor)
##LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)     ## for auto standardization

print(y_pred)


