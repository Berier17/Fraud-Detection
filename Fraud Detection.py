# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 13:46:43 2025

@author: aliel
"""
#Importing libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

#Importing the dataset
df = pd.read_csv(r'D:\creditcard.csv')
df.info()

#EDA
df.head()
df.describe()

#plot class distribution 
df['Class'].value_counts()
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0:Non Fraud, 1:Fraud')
plt.show()

#plot transaction amount
sns.histplot(df['Amount'], bins=100)
plt.title('Transaction Amount Distribution')
plt.show()

#Correlation heatmap
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

#Time vs Fraud
sns.histplot(df[df['Class'] == 1]['Time'], bins=100, color='red')
plt.title('Fraud transaction over Time')
plt.xlabel('Time')
plt.show()

#Preprocessing 
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

#Features and target
X = df.drop('Class', axis=1)
y = df['Class']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
#SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

pd.Series(y_train_sm).value_counts()

#Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_sm, y_train_sm)

#Predictions on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

#Model Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Plotting precision recall curve
precision, recall, threshold= precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision, marker='.')
plt.title('Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

#Random forest classifier model initiating and training
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')

rf.fit(X_train_sm, y_train_sm)

#Prediction on test set
rf_pred = rf.predict(X_test)

rf_proba = rf.predict_proba(X_test)[:,1]

#Model evaluation
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

#Precision recall curve 
precision, recall, threshold = precision_recall_curve(y_test, rf_proba)

plt.plot(recall, precision, marker='.', color='salmon')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve - Random Forest')
plt.show()

#Feature importance 
importances = rf.feature_importances_

#Map to feature names
features = X_train_sm.columns

#Creating DF for easier plotting
feature_importance_df = pd.DataFrame({
    'Feature':features,
    'Importance':importances
    }).sort_values(by='Importance', ascending=False)
 
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.xlabel('Importances')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importances - Random Forest')
plt.show()

#Hyperparameter tunning
param_grid= {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
    }
random_search = RandomizedSearchCV(rf, param_grid, n_iter=10, scoring='f1', cv=3, n_jobs=-1)
random_search.fit(X_train_sm, y_train_sm)
print(random_search.best_params_)

#Final random forest
final_rf = RandomForestClassifier(n_estimators=500, 
                                  min_samples_split=2,
                                  max_depth=None, 
                                  class_weight='balanced', 
                                  n_jobs=-1, 
                                  random_state=42)

final_rf.fit(X_train_sm, y_train_sm)

#Final prediction on test set
final_rf_pred = final_rf.predict(X_test)

final_rf_proba = final_rf.predict_proba(X_test)[:,1]

#mFinal model evaluation
print(confusion_matrix(y_test, final_rf_pred))
print(classification_report(y_test, final_rf_pred))

#Final precision recall curve 
precision, recall, threshold = precision_recall_curve(y_test, final_rf_proba)

plt.plot(recall, precision, marker='.', color='darkgreen')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve - Random Forest - Best Params')
plt.show()

#Saving the model
joblib.dump(final_rf, 'Fraud Detection/random_forest_model.pkl')





