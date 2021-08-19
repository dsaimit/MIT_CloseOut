"""
CASUAL LINEAR REGRESSION WITHOUT REGULARIZATION
"""
import pandas as pd
import numpy as np
import optuna

from sklearn.linear_model import LinearRegression
import sklearn
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import PoissonRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from optuna.samplers import RandomSampler

from sklearn.model_selection import cross_val_score

import timeit

import lightgbm as lgb



def calculate_quality_metrics(model_name, 
                              y_trval, 
                              y_trval_pred, 
                              y_test, 
                              y_test_pred, 
                              hyperparams, 
                              validation_mse):
    
    """
    Metrics calculation
    """
    
    results = dict()
    
    results['model_name'] = model_name
    results['hyperparameters'] = hyperparams
    results['mse_train'] = mean_squared_error(y_trval, y_trval_pred)
    results['mse_test'] = mean_squared_error(y_test, y_test_pred)
    results['mae_train'] = mean_absolute_error(y_trval, y_trval_pred)
    results['mae_test'] = mean_absolute_error(y_test, y_test_pred)
    results['metric_validation'] = validation_mse
    
    results['r2_train'] = r2_score(y_trval, y_trval_pred)
    results['r2_test'] = r2_score(y_test, y_test_pred)
    
    
    return (results,y_test_pred)
    

def linear_model(x_trainvalid, 
                 x_test, 
                 y_trainvalid, 
                 y_test):
    
    start = timeit.default_timer()

     #train model
    lm = LinearRegression(fit_intercept=True).fit(X = x_trainvalid, 
                                                  y = y_trainvalid)

    #make predictions
    y_trainvalid_pred = lm.predict(x_trainvalid)
    y_test_pred = lm.predict(x_test)

    results = calculate_quality_metrics(model_name = 'linear_model', 
                                        y_trval = y_trainvalid, 
                                        y_trval_pred = y_trainvalid_pred, 
                                        y_test = y_test, 
                                        y_test_pred = y_test_pred, 
                                        hyperparams = np.nan, 
                                        validation_mse = np.nan)

    stop = timeit.default_timer()

    print('Time for preparing linear model result: ', round((stop - start)/60.0, 2), ' minutes') 

    return([results,y_test_pred])

"""
LASSO LINEAR REGRESSION
"""

def linear_lasso(x_tr, x_te, x_trval, 
                 y_tr, y_te, y_trval, n_trials, max_alpha):
    
    start = timeit.default_timer()

    def objective_lasso(trial):
        lasso_alpha = trial.suggest_uniform('lasso_alpha', 1e-8, max_alpha)
        model = sklearn.linear_model.Lasso(alpha=lasso_alpha)

#         model.fit(x_tr, y_tr)
#         y_pred = model.predict(x_val)
        
        errors = cross_val_score(model, x_tr, y_tr, scoring = 'neg_mean_squared_error', cv = 3)

#         error = mean_squared_error(y_val, y_pred)

        error = np.mean(errors)

        return error

    study_lasso = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study_lasso.optimize(objective_lasso, n_trials=n_trials)

    best_alpha = study_lasso.best_params['lasso_alpha']

    final_model = sklearn.linear_model.Lasso(alpha=best_alpha)
    final_model.fit(x_trval, y_trval)

    y_trval_pred = final_model.predict(x_trval)
    y_test_pred = final_model.predict(x_te)
    
    results = calculate_quality_metrics(model_name = 'lasso_linear_model', 
                                        y_trval = y_trval, 
                                        y_trval_pred = y_trval_pred, 
                                        y_test = y_te, 
                                        y_test_pred = y_test_pred, 
                                        hyperparams = [study_lasso.best_params], 
                                        validation_mse = study_lasso.best_value)

    stop = timeit.default_timer()

    print('Time for preparing Lasso result: ', round((stop - start)/60.0, 2), ' minutes') 

    return([results,y_test_pred])

    """
    RIDGE LINEAR REGRESSION
    """

def linear_ridge(x_tr, x_te, x_trval, 
                 y_tr, y_te, y_trval, n_trials, max_alpha):
    
    start = timeit.default_timer()
        
    def objective_ridge(trial):
        
        ridge_alpha = trial.suggest_uniform('ridge_alpha', 1e-8, max_alpha)
        model = sklearn.linear_model.Ridge(alpha=ridge_alpha)

#         model.fit(x_tr, y_tr)
#         y_pred = model.predict(x_val)
        
        errors = cross_val_score(model, x_tr, y_tr, scoring = 'neg_mean_squared_error', cv = 3)

#         error = mean_squared_error(y_val, y_pred)

        error = np.mean(errors)
            
        return error
    
    study_ridge = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study_ridge.optimize(objective_ridge, n_trials = n_trials)

    best_alpha = study_ridge.best_params['ridge_alpha']

    final_model = sklearn.linear_model.Ridge(alpha=best_alpha)
    final_model.fit(x_trval, y_trval)

    y_trval_pred = final_model.predict(x_trval)
    y_test_pred = final_model.predict(x_te)

    results = calculate_quality_metrics(model_name = 'ridge_linear_model', 
                                        y_trval = y_trval, 
                                        y_trval_pred = y_trval_pred, 
                                        y_test = y_te, 
                                        y_test_pred = y_test_pred, 
                                        hyperparams = [study_ridge.best_params], 
                                        validation_mse = study_ridge.best_value)
    
    stop = timeit.default_timer()

    print('Time for preparing Ridge result: ', round((stop - start)/60.0, 2), ' minutes') 

    return([results,y_test_pred])

def decision_tree(x_tr, x_te, x_trval, 
                 y_tr, y_te, y_trval, n_trials):
    
    start = timeit.default_timer()
        
    def objective_decision_tree(trial):

        params = {
            'criterion': 'mse',
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'ccp_alpha': trial.suggest_uniform('ccp_alpha', 1e-8, 1000.0)
        }
        
        dt = DecisionTreeRegressor(criterion = params['criterion'],
                                   max_depth = params['max_depth'],
                                   min_samples_split = params['min_samples_split'],
                                   min_samples_leaf = params['min_samples_leaf'],
                                   ccp_alpha = params['ccp_alpha']).fit(x_tr, y_tr)
        
        errors = cross_val_score(dt, x_tr, y_tr, cv = 3)

#         preds = dt.predict(x_val)
        # multitask_masked_mean_squared_error
#         error = mean_squared_error(y_val, preds)

        error = np.mean(errors)

        return error
    
    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_decision_tree, n_trials = n_trials)
    
    best_params = {
        'criterion': 'mse',
        'max_depth': study.best_params['max_depth'],
        'min_samples_split': study.best_params['min_samples_split'],
        'min_samples_leaf': study.best_params['min_samples_leaf'],
        'ccp_alpha': study.best_params['ccp_alpha'],
    }

    final_model = DecisionTreeRegressor(criterion = best_params['criterion'],
                                       max_depth = best_params['max_depth'],
                                       min_samples_split = best_params['min_samples_split'],
                                       min_samples_leaf = best_params['min_samples_leaf'],
                                       ccp_alpha = best_params['ccp_alpha'])
    final_model.fit(x_trval, y_trval)

    y_trval_pred = final_model.predict(x_trval)
    y_test_pred = final_model.predict(x_te)

    results = calculate_quality_metrics(model_name = 'decision_tree_model', 
                                        y_trval = y_trval, 
                                        y_trval_pred = y_trval_pred, 
                                        y_test = y_te, 
                                        y_test_pred = y_test_pred, 
                                        hyperparams = [study.best_params], 
                                        validation_mse = study.best_value)
    
    stop = timeit.default_timer()

    print('Time for preparing Ridge result: ', round((stop - start)/60.0, 2), ' minutes') 

    return([results,y_test_pred])

def poisson_regression(x_tr, x_te, x_trval, 
                       y_tr, y_te, y_trval, n_trials, max_alpha):
    
    start = timeit.default_timer()
    
    tol = 1e-40

    def objective_poisson_regression(trial):
        
        alpha_pr = trial.suggest_uniform('alpha_pr', 1e-20, max_alpha)
        model = PoissonRegressor(alpha = alpha_pr, tol = tol, max_iter = 10000000)
#         , tol = 1e-10, max_iter = 10000
        
        errors = cross_val_score(model, x_tr, y_tr, scoring = 'neg_mean_squared_error', cv = 5)

#         model.fit(x_tr, y_tr)
#         y_pred = model.predict(x_val)

        error = np.mean(errors)

        return error

    study = optuna.create_study(sampler = RandomSampler(seed=1), direction='maximize')
    study.optimize(objective_poisson_regression, n_trials=n_trials)

    best_alpha = study.best_params['alpha_pr']

    final_model = PoissonRegressor(alpha = best_alpha, tol = tol, max_iter = 10000000)
    final_model.fit(x_trval, y_trval)

    y_trval_pred = final_model.predict(x_trval)
    y_test_pred = final_model.predict(x_te)

    results = calculate_quality_metrics(model_name = 'poisson_regression_model', 
                                        y_trval = y_trval, 
                                        y_trval_pred = y_trval_pred, 
                                        y_test = y_te, 
                                        y_test_pred = y_test_pred, 
                                        hyperparams = [study.best_params], 
                                        validation_mse = study.best_value)

    stop = timeit.default_timer()

    print('Time for preparing Lasso result: ', round((stop - start)/60.0, 2), ' minutes') 

    return([results,y_test_pred])


def lightgbm_model(x_tr, x_te, x_trval, 
                   y_tr, y_te, y_trval, n_trials):
    
    start = timeit.default_timer()

    def objective_lightgbm(trial):
        
        lgb_train = lgb.Dataset(x_tr, y_tr)

        param = {
         'objective': 'rmse',
         'metric': 'rmse',
         'boosting_type' : 'gbdt',
         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-20, 10.0),
         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-20, 1e-5),
         'num_leaves': trial.suggest_int('num_leaves', 100000, 100000),
         'num_boost_round': trial.suggest_int('num_boost_round', 4, 100),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 0),
        # 'min_sum_hessian_in_leaf': trial.suggest_int('min_sum_hessian_in_leaf', 0, 0),
         # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
         # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
         # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
         'max_depth': trial.suggest_int('max_depth', 2, 10),
         # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
         'learning_rate' : trial.suggest_uniform('learning_rate', 1e-5, 10.0),
         'verbosity': -1
        }

        gbm = lgb.cv(param, lgb_train, nfold = 3)

#         preds = gbm.predict(x_val, num_iteration=gbm.best_iteration)
        # multitask_masked_mean_squared_error
        
#         errors = cross_val_score(model, x_tr, y_tr, scoring = 'neg_mean_squared_error', cv = 3)

#         print(gbm)
        
        
        error = np.mean(gbm['rmse-mean'])**2

        return error

    study_lightgbm_v2 = optuna.create_study(sampler = RandomSampler(seed=1), direction='minimize')
    study_lightgbm_v2.optimize(objective_lightgbm, n_trials=n_trials)

    best_l1 = study_lightgbm_v2.best_params['lambda_l1']
    best_l2 = study_lightgbm_v2.best_params['lambda_l2']
    # best_num_leaves = study_lightgbm.best_params['num_leaves']
    best_num_boost_round = study_lightgbm_v2.best_params['num_boost_round']
    # best_min_child_samples = study_lightgbm.best_params['min_child_samples']
    best_learning_rate = study_lightgbm_v2.best_params['learning_rate']
    best_max_depth = study_lightgbm_v2.best_params['max_depth']
    best_num_leaves = study_lightgbm_v2.best_params['num_leaves']
    best_min_data_in_leaf = study_lightgbm_v2.best_params['min_data_in_leaf']
    # best_min_sum_hessian_in_leaf = study_lightgbm_v2.best_params['min_sum_hessian_in_leaf']

    param = {
     'objective': 'rmse',
     'metric': 'rmse',
     'boosting_type' : 'gbdt',
     'lambda_l1': best_l1,
     'lambda_l2': best_l2,
     'num_leaves': best_num_leaves,
     'min_data_in_leaf': best_min_data_in_leaf,
    # 'min_sum_hessian_in_leaf': best_min_sum_hessian_in_leaf,
     'num_boost_round': best_num_boost_round,
     'max_depth': best_max_depth,
    # 'min_child_samples': best_min_child_samples,
    'learning_rate' : best_learning_rate,
    }

    lgb_trainvalid = lgb.Dataset(x_trval, y_trval)
    lgb_test = lgb.Dataset(x_te, y_te, reference=lgb_trainvalid)

    final_lightgbm = lgb.train(param, lgb_trainvalid)

    y_trval_pred = final_lightgbm.predict(x_trval, num_iteration=final_lightgbm.best_iteration)
    y_test_pred = final_lightgbm.predict(x_te, num_iteration=final_lightgbm.best_iteration)
    
    results = calculate_quality_metrics(model_name = 'lightgbm_model', 
                                        y_trval = y_trval, 
                                        y_trval_pred = y_trval_pred, 
                                        y_test = y_te, 
                                        y_test_pred = y_test_pred, 
                                        hyperparams = [study_lightgbm_v2.best_params], 
                                        validation_mse = study_lightgbm_v2.best_value)

    stop = timeit.default_timer()

    print('Time for preparing LightGBM result: ', round((stop - start)/60.0, 2), ' minutes') 

    return([results,y_test_pred])


