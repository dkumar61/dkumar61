#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_experiment("RegressionSal")

# In[2]:


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[3]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


# parmeters for model
alpha = 0.7
l1_ratio = 0.7
random_state=40


# In[5]:


with mlflow.start_run():
    # Fitting Simple Linear Regression to the Training set
    #from sklearn.linear_model import LinearRegression
    
    #regressor = LinearRegression()
    #regressor.fit(X_train, y_train)
    
    from sklearn.linear_model import ElasticNet
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(X_train, y_train)
    
    # Predicting the Test set results
    #y_pred = regressor.predict(X_test)
    y_pred = lr.predict(X_test)
    #np.savetxt("result.csv", y_pred, delimiter=",")

    from sklearn import metrics

    mlflow.log_metric("MAE", metrics.mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("MSE", metrics.mean_squared_error(y_test, y_pred))
    mlflow.log_metric("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    mlflow.log_metric("r2", metrics.r2_score(y_test, y_pred))
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1.to_csv('result.txt', encoding='utf-8', index=False)
    
    mlflow.log_artifact("Salary_Data.csv")
    mlflow.log_artifact("result.txt")
    
    accuracy = lr.score(X_test,y_test)
    print(accuracy*100,'%')
    mlflow.log_param("accuracy", accuracy)
    
    mlflow.sklearn.log_model(lr, "model")


# In[6]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[7]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[8]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[10]:


mlflow.search_runs()


# In[10]:


run_id1 = "b3228b3a4a6d45e89e6ce98a3453a111"
model_uri = "file:///Users/dhananjay/Desktop/Machine%20Learning/Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/mlruns/3/b3228b3a4a6d45e89e6ce98a3453a111/artifacts/model"


# In[11]:


model = mlflow.sklearn.load_model(model_uri = model_uri)


# In[12]:


model.get_params()


# In[13]:


#import numpy as np
#arr = np.array([[20]])
#model.predict(arr)
model.predict(X_test)

