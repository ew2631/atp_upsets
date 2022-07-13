# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from model_results import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import preprocessing, linear_model, model_selection, metrics, datasets, base
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    df = df.sort_values("Date")
    df["WRank"] = df["WRank"].replace(np.nan, 0)
    df["WRank"] = df["WRank"].replace("NR", 2000)
    df["LRank"] = df["LRank"].replace(np.nan, 0)
    df["LRank"] = df["LRank"].replace("NR", 2000)
    df["WRank"] = df["WRank"].astype(int)
    df["LRank"] = df["LRank"].astype(int)
    df["Wsets"] = df["Wsets"].astype(float)
    df['Wsets'] = df['Wsets'].replace(np.nan, 0.0)
    df["Lsets"] = df["Lsets"].replace("`1", 1)
    df["Lsets"] = df["Lsets"].astype(float)
    df['Lsets'] = df['Lsets'].replace(np.nan, 0.0)
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(x), '%m/%d/%Y'))
    df['Year'] = df['Date'].apply(lambda x: x.year)
    # new var: Rank_Delta
    df['Rank_Delta'] = df['WRank'] - df['LRank']
    df = df.sort_values("Date")
    df.reset_index(drop=True, inplace=True)
    return df

def upset(row):
    if row['Rank_Delta']>0:
        return 1
    else:
        return 0

df = pd.read_csv(r"C:\Users\emwang\metis bootcamp\atp_data.csv")
df=clean_data(df)

################### Feature "engineering" ################

df['upset']=df.apply(lambda row: upset(row),axis=1)
target_var=df['upset']
cat_vars = ["Series","Court","Surface","Round","Tournament",'Comment']
cat_features=pd.get_dummies(df[cat_vars], drop_first=True)
numeric_vars=['Wsets','Lsets','Year','elo_winner','elo_loser']
numeric_features=df[numeric_vars]
all_features=pd.concat([numeric_features,cat_features],1)

print('Percent Upsets: {a}'.format(a=len(df[df['upset']==1])/len(df)))
print('Percent Expected: {a}'.format(a=len(df[df['upset']==0])/len(df)))


################### Train Test Split ################

X_train, X_test, y_train, y_test = train_test_split(all_features, target_var, test_size=0.20, random_state=42)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train),columns=list(X_train.columns))
X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=list(X_train.columns))

################### Logistic Regression ################

#Establish baseline logistic regression model
lr = LogisticRegression(solver='liblinear', penalty = 'l2',random_state=42)
lr.fit(X_train_scaled, y_train)
lr_predictions=lr.predict(X_test_scaled)
#
# #Understanding model outcomes
lr_results=final_comparison([lr], X_test, y_test)
print(lr_results)

cm_plt=make_confusion_matrix(lr, X_test_scaled, y_test,class_labels=['not upset','upset'])
cm_plt.show()
roc_curve=plot_roc(lr, X_test, y_test)
roc_curve.show()

coef_table = pd.DataFrame(list(X_train_scaled.columns),columns=['Variable']).copy()
coef_table.insert(len(coef_table.columns),"Coefs",lr.coef_.transpose())
coefficient_table= pd.concat([coef_table.sort_values('Coefs', ascending=False)[:5], coef_table.sort_values('Coefs', ascending=False)[-5:]])
print(coefficient_table)
coefficient_table.to_csv('coefficient_table.csv')

################### Decision Tree ################

#For Decision Trees, going to tune hyperparameters to get best model performance
check_params={
                'max_leaf_nodes': list(range(1000,11000,1000)),
                'criterion': ['gini','entropy'],
                'max_depth': np.arange(50 ,100,5),
                'min_impurity_decrease': np.arange(0.0,0.1,0.01)
            }
clf = tree.DecisionTreeClassifier(random_state=65)
clf.fit(X_train, y_train)
create_grid=GridSearchCV(clf, param_grid=check_params, cv=4, verbose=10, return_train_score=True, scoring='roc_auc')
create_grid.fit(X_train, y_train)
print("Train score for %d fold CV := %3.2f" %(4, create_grid.score(X_train, y_train)))
print("Test score for %d fold CV := %3.2f" %(4, create_grid.score(X_test, y_test)))
print ("!!!! best fit parameters from GridSearchCV !!!!")
print (create_grid.best_params_)
#
# #save grid search results in case we need to debug
joblib.dump(create_grid, 'grid_search_clf.pkl')
#
# #WINNING MODEL
clf = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=50, max_leaf_nodes= 1000, min_impurity_decrease=0.01, random_state=65)
clf.fit(X_train, y_train)
predictions=clf.predict(X_test)
print(classification_report(y_test, predictions))
print(roc_auc_score(y_test, predictions))
score=metrics.accuracy_score(y_test,predictions)
precision=metrics.precision_score(y_test,predictions)
recall=metrics.recall_score(y_test,predictions)

clf_results=final_comparison([clf], X_test, y_test)
print(clf_results)

cm_plt=make_confusion_matrix(clf, X_test, y_test,class_labels=['not upset','upset'])
cm_plt.show()
roc_curve=plot_roc(clf, X_test, y_test)
roc_curve.show()

# ################### Random Forest ###################

# Valid parameters:
# ['bootstrap', 'ccp_alpha', 'class_weight', 'criterion',
# 'max_depth', 'max_features', 'max_leaf_nodes',
# 'max_samples', 'min_impurity_decrease', 'min_samples_leaf',
# 'min_samples_split', 'min_weight_fraction_leaf',
# 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start'].

check_params={
                'class_weight': ["balanced", {0: 1, 1: 2}]
                ,'n_estimators': list(range(200, 1100,100))
                ,'max_features': ['sqrt', 'log2', None]
                #'max_depth': np.arange(10, 60, 10),
                #'min_impurity_decrease': np.arange(0.0,0.1,0.01),
                #'criterion': ['mse','mae'],
            }
rf = RandomForestClassifier(n_jobs=-1, random_state=65)

create_grid=GridSearchCV(rf, param_grid=check_params, cv=4, verbose=10, return_train_score=True, scoring='roc_auc')
create_grid.fit(X_train, y_train)
print("Train score for %d fold CV := %3.2f" %(4, create_grid.score(X_train, y_train)))
#print("Validation score for %d fold CV := %3.2f" %(4, create_grid.score(X_val, y_val)))
print ("!!!! best fit parameters from GridSearchCV !!!!")
print (create_grid.best_params_)

print(create_grid.cv_results_)

#save your model or results
joblib.dump(create_grid, 'grid_search_rf.pkl')

#WINNING MODEL
rf = RandomForestClassifier(class_weight= {0: 1, 1: 2}, max_features= None, n_estimators= 900, random_state=65)
rf.fit(X_train, y_train)

rf_results=final_comparison([rf], X_test, y_test)
print(rf_results)
cm_plt=make_confusion_matrix(rf, X_test, y_test,class_labels=['not upset','upset'])
cm_plt.show()
roc_curve=plot_roc(rf, X_test, y_test)
roc_curve.show()
#
# ################### XGBoost ###################
#
#
xgb_model = xgb.XGBClassifier(random_state = 89)
xgb_model.fit(X_train,y_train)
xgb_params={
                'n_estimators': list(range(1000, 2000,500))
                ,'max_depth': np.arange(10,12)
                , 'eta': [0.0,0.5,1]
                , 'subsample': [0.5, 1]
                , 'colsample_bytree': [0.5, 1]
             }
create_grid=GridSearchCV(xgb_model, param_grid=xgb_params, cv=4, verbose=10, return_train_score=True, scoring='roc_auc')
create_grid.fit(X_train, y_train)
print("Train score for %d fold CV := %3.2f" %(4, create_grid.score(X_train, y_train)))
print("Validation score for %d fold CV := %3.2f" %(4, create_grid.score(X_val, y_val)))
print ("!!!! best fit parameters from GridSearchCV !!!!")
print (create_grid.best_params_)
joblib.dump(create_grid, 'grid_search_xgb.pkl')
#
# #WINNING MODEL
xgb_model = xgb.XGBClassifier(
     max_depth=10
    ,random_state=89
    ,verbosity=3
)
xgb_model.fit(X_train, y_train)

xgb_results=final_comparison([xgb_model], X_test, y_test)
print(xgb_results)
cm_plt=make_confusion_matrix(xgb_model, X_test, y_test,class_labels=['not upset','upset'])
cm_plt.show()
roc_curve=plot_roc(xgb_model, X_test, y_test)
roc_curve.show()

# ################### Model Comparison ###################

final_scores = final_comparison([clf, rf,xgb_model], X_test, y_test)
final_scores.columns=['Decision Tree', 'Random Forest','XGB']
print(final_scores)
final_scores.to_csv('model_baseline.csv')
################### END OF CODE ###################
