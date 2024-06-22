#!/usr/bin/env python
# coding: utf-8

# QUESTION 1

# In[8]:


import sqlite3
import numpy as np
import pandas as pd

def DBconnect():#Connection to database
    conn = sqlite3.connect("heart_db.db")
    return conn

def SQLTable(conn):#Creating table
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS patience")
    
    df = pd.read_csv('C:/Users/Thobani Lukhele/Desktop/EDUVOS/BSCHons IT/First Semester/Data Mining and Data Administration/Project1 Assignment1/heart.csv',header = 0,sep=";")
    data = """
    CREATE TABLE patience(
        age CHAR(100),
        sex CHAR(100),
        cp CHAR(10),
        trestbps int,
        chol int,
        fbs int,
        restecg int,
        thalach int,
        exang int,
        oldpeak float(10),
        slope int,
        ca int,
        thal int,
        target int
        )"""

    cur.execute(data)
    df.to_sql('patience', conn, if_exists='append', index=False)
    
   # print("Database has been created")
    conn.commit()
    #print(df)
   # print('\n Records added')
    
def DB_Read(conn):
    cur = conn.cursor()
    data = pd.read_sql("SELECT * FROM  patience;",conn)
    conn.commit()
    #print(data)

conn = DBconnect()
SQLTable(conn)
DB_Read(conn)

conn.close()


# QUESTION 2.1

# A

# In[9]:


import sqlite3
import numpy as np
import pandas as pd

def DBconnect():#Connection to database
    conn = sqlite3.connect("heart_db.db")
    return conn

def SQLTable(conn):#Creating table
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS patience")
    
    df = pd.read_csv('C:/Users/Thobani Lukhele/Desktop/EDUVOS/BSCHons IT/First Semester/Data Mining and Data Administration/Project1 Assignment1/heart.csv',header = 0,sep=";")
    
    data = """
    CREATE TABLE patience(
        age CHAR(100),
        sex CHAR(100),
        cp CHAR(10),
        trestbps int,
        chol int,
        fbs int,
        restecg int,
        thalach int,
        exang int,
        oldpeak float(10),
        slope int,
        ca int,
        thal int,
        target int
        )"""

    cur.execute(data)
    df.to_sql('patience', conn, if_exists='append', index=False) 
    
   # print("Database has been created")
    conn.commit()
   # print('\n Records added')
    
def DB_Read(conn):
    cur = conn.cursor()
    data = pd.read_sql("SELECT * FROM  patience;",conn)
    conn.commit()
    #print(data)
    
        #QUESTION 2.1 (A)
    
#     sql_query = """ SELECT * FROM patience;"""

#     df = pd.read_sql_query(sql_query, conn)
    
#     df.drop_duplicates(inplace=True)
    
#     print(df.loc[df.duplicated()])   #THIS CODE IS NOT COMPULSORY
    
#     print(df.shape) # THIS CODE IS NOT COMPULSORY
    
#     df.to_sql('patience', conn, if_exists='replace', index=False) # This will go to the script but you cannot run this code cause the table has not yet been created
    
#     conn.commit()
    #End Here
    
    data = pd.read_sql("SELECT * FROM  patience;",conn)
    conn.commit()
    #print(data)

conn = DBconnect()
SQLTable(conn)
DB_Read(conn)

#conn.close()


# QUESTION 2.1 (B) & (C) I JUST NEED TO CHANGE THE COLUMNS

# In[10]:


# QUESTION 2.1 B

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establish a connection to the SQLite database
def DBconnect():#Connection to database
    conn = sqlite3.connect("heart_db.db")
    return conn

# Define the SQL query to fetch the data including the target variable and categorical variables
sql_query = """
SELECT sex, cp, fbs, restecg, exang, slope, ca, thal, target
FROM patience;
"""

# Load the data into a pandas DataFrame
df = pd.read_sql_query(sql_query, conn)

# Close the connection
#conn.close()

# Plot the distribution of classes for each categorical variable based on the target variable
target_variable = 'target'

categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']

# Set up the plot grid
num_plots = len(categorical_variables)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

# Plot the distribution of classes for each categorical variable
for i, var in enumerate(categorical_variables):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    sns.countplot(x=var, hue=target_variable, data=df, ax=ax)
    ax.set_title(f'Distribution of {var} by {target_variable}')
    ax.set_xlabel(var)
    ax.set_ylabel('Count')
    ax.legend(title=target_variable, loc='upper right')

# Adjust layout
plt.tight_layout()
plt.show()

# This code will plot the distribution of classes for each categorical variable based on the target variable. Observations that can be derived from these plots include:

# Class Imbalance: Check if there is a significant class imbalance in any of the categorical variables, which might affect the performance of predictive models.
# Relationship between Variables: Explore how the distribution of classes in each categorical variable varies with the target variable. This can provide insights into the relationship between variables and potential predictive power.
# Data Quality Issues: Identify any irregularities or anomalies in the data distribution that may indicate data quality issues or errors.
# Predictive Power: Assess the potential predictive power of each categorical variable based on the distribution of classes relative to the target variable.



# In[11]:


# QUESTION 2.1 C

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establish a connection to the SQLite database
def DBconnect():#Connection to database
    conn = sqlite3.connect("heart_db.db")
    return conn

# Define the SQL query to fetch the data including the target variable and categorical variables
sql_query = """
SELECT age, trestbps, chol, thalach, oldpeak, target
FROM patience;
"""

# Load the data into a pandas DataFrame
df = pd.read_sql_query(sql_query, conn)

# Close the connection
conn.close()

# Plot the distribution of classes for each categorical variable based on the target variable
target_variable = 'target'
numeric_variables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# Plot the distribution of classes for each numeric variable based on the target variable
for num_var in numeric_variables:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=num_var, hue=target_variable, kde=True, bins=20)
    plt.title(f'Distribution of {num_var} based on {target_variable}')
    plt.xlabel(num_var)
    plt.ylabel('Count')
    plt.legend(title=target_variable)
    plt.show()
    
    
    
    
# Observations that can be derived from these plots:

# Distribution Separation: Check if there is a noticeable separation in the distributions of numeric variables between different classes of the target variable. This can indicate the potential predictive power of these variables.
# Outlier Detection: Look for outliers in the numeric variables for each class of the target variable. Outliers may have a significant impact on model performance and should be treated appropriately during preprocessing.
# Skewness and Kurtosis: Observe the skewness and kurtosis of the distributions. High skewness or kurtosis may indicate non-normality in the data, which may require transformation before modeling.
# Data Range: Compare the ranges of numeric variables across different classes of the target variable. Significant differences in data ranges may suggest different behaviors or characteristics for each class.
# Data Overlaps: Identify areas of overlap in the distributions of numeric variables between different classes of the target variable. Overlapping regions may indicate ambiguity or uncertainty in classification boundaries and may require further feature engineering or modeling techniques to improve predictive performance.


# # Set up the plot grid
# num_plots = len(categorical_variables)
# num_cols = 3
# num_rows = (num_plots + num_cols - 1) // num_cols
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

# # Plot the distribution of classes for each categorical variable
# for i, var in enumerate(categorical_variables):
#     row = i // num_cols
#     col = i % num_cols
#     ax = axes[row, col]
#     sns.countplot(x=var, hue=target_variable, data=df, ax=ax)
#     ax.set_title(f'Distribution of {var} by {target_variable}')
#     ax.set_xlabel(var)
#     ax.set_ylabel('Count')
#     ax.legend(title=target_variable, loc='upper right')

# # Adjust layout
# plt.tight_layout()
# plt.show()


# QUESTION 3.1

# In[12]:


import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def DBconnect():#Connection to database
    conn = sqlite3.connect("heart_db.db")
    return conn

def SQLTable(conn):#Creating table
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS patience")
    
    df = pd.read_csv('C:/Users/Thobani Lukhele/Desktop/EDUVOS/BSCHons IT/First Semester/Data Mining and Data Administration/Project1 Assignment1/heart.csv',header = 0,sep=";")
    
    data = """
    CREATE TABLE patience(
        age CHAR(100),
        sex CHAR(100),
        cp CHAR(10),
        trestbps int,
        chol int,
        fbs int,
        restecg int,
        thalach int,
        exang int,
        oldpeak float(10),
        slope int,
        ca int,
        thal int,
        target int
        )"""

    cur.execute(data)
    df.to_sql('patience', conn, if_exists='append', index=False)
    conn.commit()
    
def DB_Read(conn):
    cur = conn.cursor()
    data = pd.read_sql("SELECT * FROM  patience;",conn)
    conn.commit()
    
    #Cleaning the data by removing duplicates
    sql_query = """ SELECT * FROM patience;"""

    df = pd.read_sql_query(sql_query, conn)
    
    df.drop_duplicates(inplace=True)
    
    df.to_sql('patience', conn, if_exists='replace', index=False) # This will go to the script but you cannot run this code cause the table has not yet been created
    
    conn.commit()
    
    data = pd.read_sql("SELECT * FROM  patience;",conn)
    conn.commit()

conn = DBconnect()
SQLTable(conn)
DB_Read(conn)

# Encoding categorical variables (if any)
# For simplicity, let's assume there are no categorical variables in this example
data = pd.read_sql("SELECT * FROM  patience;",conn)

# Scaling numerical features
scaler = StandardScaler()
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 3: Exploratory Data Analysis (EDA)
# Let's print basic statistics and examine the first few rows of the data
print(data.describe())
print(data.head())

# Step 4: Feature Selection
# Let's select all features except the target variable 'target' for now
features = data.drop('target', axis=1)

# Target variable
target = data['target']

# Step 5: Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

conn.close()


# QUESTION 3.2

# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Initialize models
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()
svm = SVC()

# Fit models to the training data
logistic_regression.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Perform predictions
lr_pred = logistic_regression.predict(X_test)
rf_pred = random_forest.predict(X_test)
svm_pred = svm.predict(X_test)

# Evaluate performance
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Print accuracies
print("Logistic Regression Accuracy:", lr_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("SVM Accuracy:", svm_accuracy)

# Save the best performing model to disk
best_model = max([(lr_accuracy, logistic_regression), (rf_accuracy, random_forest), (svm_accuracy, svm)], key=lambda x: x[0])[1]
joblib.dump(best_model, 'best_model.pkl')


# In[ ]:




