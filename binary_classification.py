#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Scientific
import numpy as np
import pandas as pd
from scipy import stats
from scipy import interp

# General
import itertools
import copy
import re
from datetime import datetime
import random
import timeit
from itertools import cycle


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML
## Optimizing Models
import optuna
import sklearn
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report

## Actual Models 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb

from sklearn.metrics import accuracy_score


# Optuna
import optuna
from optuna.samplers import RandomSampler


# In[10]:


def calculate_quality_metrics(model_name, 
                              trval_y, 
                              y_trval_pred, 
                              test_y, 
                              y_test_pred,
                              hyperparameters):
    
    """
    Metrics calculation
    """

    y_trval_proba = y_trval_pred[0]
    y_trval_labels = y_trval_pred[1]
    
    y_test_proba = y_test_pred[0]
    y_test_labels = y_test_pred[1]
    
    results = dict()
    
    results['model_name'] = model_name
    results['hyperparameters'] = hyperparameters
    results['roc_auc_eval'] = roc_auc_score(trval_y, y_trval_proba[:,1])#, multi_class = 'ovr', average = 'macro')
    results['roc_auc_test'] = roc_auc_score(test_y, y_test_proba[:,1])#, multi_class = 'ovr', average = 'macro')
        
#     get_roc_auc_plots(y_test = trval_y, 
#                       y_score = y_trval_proba, 
#                       model = model_name, 
#                       plots_for = f'{model_name}_Validation_Set')
    
#     get_roc_auc_plots(y_test = test_y, 
#                       y_score = y_test_proba,
#                       model = model_name, 
#                       plots_for = f'{model_name}_Test_Set')
        
    results['f1_eval'] = f1_score(trval_y, y_trval_labels)
    results['f1_test'] = f1_score(test_y, y_test_labels)
    results['acc_eval'] = accuracy_score(trval_y, y_trval_labels)
    results['acc_test'] = accuracy_score(test_y, y_test_labels)
        
    return results


# In[11]:


def get_roc_auc_plots(y_test, y_score, model, plots_for):
    
    print(f'Getting plots for {plots_for}')
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    n_classes = 1
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, 1], y_score[:, 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    #plt.savefig(f'{plots_for}_ROC_AUC.png')
    
    pass

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# In[12]:


# Logistic Regression
def logistic_model(train_X, eval_X, trval_X, test_X, 
                   train_y, eval_y, trval_y, test_y, 
                   n_trials):
    
    start = timeit.default_timer()
    
    def objective_logistic(trial):
        
        params = {'penalty': trial.suggest_categorical('penalty', ['l1']),
                  'C': trial.suggest_uniform('C', 0, 100),
                  'multi_class': trial.suggest_categorical('multi_class',['auto']),
                  'solver': trial.suggest_categorical('solver',['liblinear']),
                  'max_iter': trial.suggest_categorical('max_iter',[10000])}
        
        logReg = LogisticRegression(**params).fit(train_X, train_y)
    
        errors = model_selection.cross_val_score(logReg, train_X, train_y, cv=3, scoring = 'roc_auc')
        error = np.mean(errors)
        
        return error

    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_logistic, n_trials = n_trials)
    
    final_model = LogisticRegression(**study.best_params)
    final_model.fit(trval_X, trval_y)

    
    # Iterate over range of thresholds to find the best alpha for f1 score 
    # Fit model using different thresholds (iterrange)
    
    # Take best alpha and uses it in testing
    
    thresholds = np.arange(0, 1, 0.001)
    y_trval_pred_proba = final_model.predict_proba(trval_X)
    f1_scores = [f1_score(trval_y, to_labels(y_trval_pred_proba[:,1], t)) for t in thresholds]
    ix = np.argmax(f1_scores)
    y_trval_pred_vals = to_labels(y_trval_pred_proba[:,1], thresholds[ix])
    
#     y_trval_pred_vals = final_model.predict(trval_X)
#     y_trval_pred_vals = convert_to_labels(y_trval_pred_vals, definitions)
    y_trval_pred = [y_trval_pred_proba, y_trval_pred_vals]
    
    y_test_pred_proba = final_model.predict_proba(test_X)
    y_test_pred_vals = to_labels(y_test_pred_proba[:,1], thresholds[ix])
#     y_test_pred_vals = final_model.predict(test_X)

#     y_test_pred_vals = convert_to_labels(y_test_pred_vals, definitions)
    y_test_pred = [y_test_pred_proba, y_test_pred_vals]

    
    results = calculate_quality_metrics(model_name = 'logistic', 
                                        trval_y = trval_y, 
                                        y_trval_pred = y_trval_pred, 
                                        test_y = test_y, 
                                        y_test_pred = y_test_pred,
                                        hyperparameters = [study.best_params])

    stop = timeit.default_timer()

    print(f'Time for preparing Logistic model result: ', round((stop - start)/60.0, 2), ' minutes') 

    return results


# In[13]:


# Decision Tree
def decision_tree(train_X, eval_X, trval_X, test_X, 
                  train_y, eval_y, trval_y, test_y, 
                  n_trials):
    
    start = timeit.default_timer()
        
    def objective_decision_tree(trial):

        params = {
            'criterion': trial.suggest_categorical('criterion',['gini']),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'ccp_alpha': trial.suggest_uniform('ccp_alpha', 1e-1, 10.0)
        }
        
        dt = DecisionTreeClassifier(**params).fit(train_X, train_y)
        
        errors = cross_val_score(dt, train_X, train_y, cv = 3, scoring = 'roc_auc')
        error = np.mean(errors)

        return error
    
    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_decision_tree, n_trials = n_trials)

    final_model = DecisionTreeClassifier(**study.best_params)
    final_model.fit(trval_X, trval_y)

    thresholds = np.arange(0, 1, 0.001)
    y_trval_pred_proba = final_model.predict_proba(trval_X)
    f1_scores = [f1_score(trval_y, to_labels(y_trval_pred_proba[:,1], t)) for t in thresholds]
    ix = np.argmax(f1_scores)
    y_trval_pred_vals = to_labels(y_trval_pred_proba[:,1], thresholds[ix])
    
#     y_trval_pred_vals = final_model.predict(trval_X)
#     y_trval_pred_vals = convert_to_labels(y_trval_pred_vals, definitions)
    y_trval_pred = [y_trval_pred_proba, y_trval_pred_vals]
    
    y_test_pred_proba = final_model.predict_proba(test_X)
    y_test_pred_vals = to_labels(y_test_pred_proba[:,1], thresholds[ix])
#     y_test_pred_vals = final_model.predict(test_X)

#     y_test_pred_vals = convert_to_labels(y_test_pred_vals, definitions)
    y_test_pred = [y_test_pred_proba, y_test_pred_vals]

    results = calculate_quality_metrics(model_name = 'decision_tree_model', 
                                        trval_y = trval_y, 
                                        y_trval_pred = y_trval_pred, 
                                        test_y = test_y, 
                                        y_test_pred = y_test_pred,
                                        hyperparameters = [study.best_params])
    
    stop = timeit.default_timer()

    print(f'Time for preparing Decision Tree result: ', round((stop - start)/60.0, 2), ' minutes') 

    return results


# In[14]:


def rf_model(train_X, eval_X, trval_X, test_X, 
             train_y, eval_y, trval_y, test_y, 
             n_trials):
    
    start = timeit.default_timer()
    
    def objective_rf(trial):
        
        params = {'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'n_estimators': trial.suggest_int('n_estimators', 3, 20)}
        
        rf = RandomForestClassifier(**params).fit(train_X, train_y)
        
        errors = cross_val_score(rf, train_X, train_y, cv = 3, scoring = 'roc_auc')
        error = errors.mean()
        
        return error
    
    
    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_rf, n_trials=n_trials)
    
    final_model = RandomForestClassifier(**study.best_params)
    final_model.fit(trval_X, trval_y)

    thresholds = np.arange(0, 1, 0.001)
    y_trval_pred_proba = final_model.predict_proba(trval_X)
    f1_scores = [f1_score(trval_y, to_labels(y_trval_pred_proba[:,1], t)) for t in thresholds]
    ix = np.argmax(f1_scores)
    y_trval_pred_vals = to_labels(y_trval_pred_proba[:,1], thresholds[ix])
    
#     y_trval_pred_vals = final_model.predict(trval_X)
#     y_trval_pred_vals = convert_to_labels(y_trval_pred_vals, definitions)
    y_trval_pred = [y_trval_pred_proba, y_trval_pred_vals]
    
    y_test_pred_proba = final_model.predict_proba(test_X)
    y_test_pred_vals = to_labels(y_test_pred_proba[:,1], thresholds[ix])
#     y_test_pred_vals = final_model.predict(test_X)

#     y_test_pred_vals = convert_to_labels(y_test_pred_vals, definitions)
    y_test_pred = [y_test_pred_proba, y_test_pred_vals]
    
    
    results = calculate_quality_metrics(model_name = 'rf_model',  
                                        trval_y = trval_y, 
                                        y_trval_pred = y_trval_pred, 
                                        test_y = test_y, 
                                        y_test_pred = y_test_pred,
                                        hyperparameters = [study.best_params])

    stop = timeit.default_timer()

    print('Time for preparing Random Forest result: ', round((stop - start)/60.0, 2), ' minutes')

    return results


# In[15]:


def xgb_model(train_X, eval_X, trval_X, test_X, 
              train_y, eval_y, trval_y, test_y, 
              n_trials):
    
    start = timeit.default_timer()

    def objective_xgb(trial):
       
        param = {
                 'lambda': trial.suggest_loguniform('lambda', 1e-5, 10.0),
                 'alpha': trial.suggest_loguniform('alpha', 1e-5, 1.0),
                 'subsample': trial.suggest_categorical('subsample', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                 'learning_rate' : trial.suggest_uniform('learning_rate', 1, 10.0),
                 'n_estimators': trial.suggest_int('n_estimators', 5, 25),
                 'max_depth': trial.suggest_int('max_depth', 4, 20)
                }

        xgb_model = xgb.XGBClassifier(**param)
        xgb_model.fit(train_X, train_y)
        
        errors = cross_val_score(xgb_model, train_X, train_y, cv = 3, scoring = 'roc_auc')
        error = errors.mean()
        return error
    
        
    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_xgb, n_trials=n_trials)
    
    final_model = xgb.XGBClassifier(**study.best_params)          
    final_model.fit(trval_X, trval_y)

    thresholds = np.arange(0, 1, 0.001)
    y_trval_pred_proba = final_model.predict_proba(trval_X)
    f1_scores = [f1_score(trval_y, to_labels(y_trval_pred_proba[:,1], t)) for t in thresholds]
    ix = np.argmax(f1_scores)
    y_trval_pred_vals = to_labels(y_trval_pred_proba[:,1], thresholds[ix])
    
#     y_trval_pred_vals = final_model.predict(trval_X)
#     y_trval_pred_vals = convert_to_labels(y_trval_pred_vals, definitions)
    y_trval_pred = [y_trval_pred_proba, y_trval_pred_vals]
    
    y_test_pred_proba = final_model.predict_proba(test_X)
    y_test_pred_vals = to_labels(y_test_pred_proba[:,1], thresholds[ix])
#     y_test_pred_vals = final_model.predict(test_X)

#     y_test_pred_vals = convert_to_labels(y_test_pred_vals, definitions)
    y_test_pred = [y_test_pred_proba, y_test_pred_vals]
    

    results = calculate_quality_metrics(model_name = 'xgb_model', 
                                        trval_y = trval_y, 
                                        y_trval_pred = y_trval_pred, 
                                        test_y = test_y, 
                                        y_test_pred = y_test_pred,
                                        hyperparameters = [study.best_params])

    stop = timeit.default_timer()

    print('Time for preparing XGB result: ', round((stop - start)/60.0, 2), ' minutes')

    return results


# In[16]:


def get_cat_idx(df, cat_cols):
    '''
    Input: list of categorical features (cat_cols), and dataframe (df)
    Output: indexes of categorical features
    '''
    cat_feat_idxs = [df.columns.get_loc(c) for c in cat_cols if c in df]
    return cat_feat_idxs


# In[17]:


def cb_model(train_X, eval_X, trval_X, test_X, 
             train_y, eval_y, trval_y, test_y, 
             n_trials, cat_feat_idxs):
    
    train_pool = cb.Pool(data = train_X, label = train_y, cat_features = cat_feat_idxs)
    eval_pool = cb.Pool(data = eval_X, label = eval_y, cat_features = cat_feat_idxs)
    trval_pool = cb.Pool(data = trval_X, label = trval_y, cat_features = cat_feat_idxs)
    test_pool = cb.Pool(data = test_X, label = test_y.values, cat_features = cat_feat_idxs)
    
    start = timeit.default_timer()
    
    def objective_cb(trial):
        params = {"iterations": trial.suggest_categorical('iterations',[2500]), #CHANGED
                  "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.15),
                  "depth": trial.suggest_int("depth", 3, 15),
                  "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                  "bootstrap_type": trial.suggest_categorical(
                  "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                  "loss_function": trial.suggest_categorical('loss_function',['CrossEntropy']),
                  "learning_rate": trial.suggest_uniform("learning_rate",0.5, 2.0),
                  "random_strength": trial.suggest_uniform("random_strength", 0.5, 2.0),
                  "leaf_estimation_method": trial.suggest_categorical('leaf_estimation_method',['Newton']),
                  "eval_metric": trial.suggest_categorical('eval_metric',['AUC'])
                 }

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = cb.CatBoostClassifier(**params)
        model.fit(train_pool, eval_set = eval_pool, verbose=500) 

#         if using_AUC:
            # Macro ROC-AUC Score
        preds = model.predict_proba(eval_X)[:,1]
        score = roc_auc_score(eval_y, preds)

#         elif not using_AUC:
#             # Macro F1 
#             preds = model.predict(eval_X)
#             score = f1_score(eval_y, preds)
        return score


    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_cb, n_trials = n_trials)

    final_model = cb.CatBoostClassifier(**study.best_params)
    final_model.fit(trval_pool)

    
    thresholds = np.arange(0, 1, 0.001)
    y_trval_pred_proba = final_model.predict_proba(trval_X)
    f1_scores = [f1_score(trval_y, to_labels(y_trval_pred_proba[:,1], t)) for t in thresholds]
    ix = np.argmax(f1_scores)
    y_trval_pred_vals = to_labels(y_trval_pred_proba[:,1], thresholds[ix])
    
#     y_trval_pred_vals = final_model.predict(trval_X)
#     y_trval_pred_vals = convert_to_labels(y_trval_pred_vals, definitions)
    y_trval_pred = [y_trval_pred_proba, y_trval_pred_vals]
    
    y_test_pred_proba = final_model.predict_proba(test_X)
    y_test_pred_vals = to_labels(y_test_pred_proba[:,1], thresholds[ix])
#     y_test_pred_vals = final_model.predict(test_X)

#     y_test_pred_vals = convert_to_labels(y_test_pred_vals, definitions)
    y_test_pred = [y_test_pred_proba, y_test_pred_vals]
    
    
    results = calculate_quality_metrics(model_name = 'cb_model', 
                                        trval_y = trval_y, 
                                        y_trval_pred = y_trval_pred, 
                                        test_y = test_y, 
                                        y_test_pred = y_test_pred,
                                        hyperparameters = [study.best_params])

    stop = timeit.default_timer()

    print(f'Time for preparing CatBoost result: ', round((stop - start)/60.0, 2), ' minutes') 

    return results


# In[ ]:


def get_x_y_split(df, y_col = 'Ontime'):
    '''
    Simply splits the dataframe into the dependent and independent variables
    '''
    
    df_y = df.loc[:, y_col]
    df_x = df.loc[:, df.columns != y_col]

    return df_x, df_y


# In[ ]:


def split_data(df, DATE_BOUND_TRAIN, DATE_BOUND_EVAL, DATE_BOUND_TEST):
    '''
    Input: processed dataframe for a specific machine learning model + boundary dates 
    Output: the train/validation/test splits 
    '''
    
    col = 'DBL_ActualDate'
    
    # Training set
    train = df[df.loc[:,col] < DATE_BOUND_TRAIN]
    train = train.drop([col], axis=1)
    train_X, train_y = get_x_y_split(train)

    # Eval set
    eval_set = df[(df.loc[:,col] >= DATE_BOUND_TRAIN) & (df.loc[:,col] < DATE_BOUND_EVAL)]
    eval_set = eval_set.drop([col], axis=1)
    eval_X, eval_y = get_x_y_split(eval_set)

    #Training + Validation
    trval_X = pd.concat([train_X,eval_X])
    trval_y = pd.concat([train_y,eval_y])
    
    # Testing set
    test = df[(df.loc[:,col] >= DATE_BOUND_EVAL) & (df.loc[:,col] < DATE_BOUND_TEST)]
    test = test.drop([col], axis=1)
    test_X, test_y = get_x_y_split(test)
    
    return train_X, eval_X, trval_X, test_X, train_y, eval_y, trval_y, test_y

