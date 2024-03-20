#!/usr/bin/env python
# coding: utf-8

# # **LIBRARIES**

# In[1]:


import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
# import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from scipy.stats import kurtosis, iqr, skew
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from glob import glob
from pathlib import Path
from datetime import datetime
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve, auc
from tqdm.notebook import tqdm
import joblib
import lightgbm as lgb
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# # **CONFIGURATION**

# In[39]:


# GENERAL CONFIGURATIONS
NUM_THREADS = 4
DATA_DIRECTORY = "./parquet_files/"
# DATA_DIRECTORY = "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/"
SUBMISSION_SUFIX = "_model2_0"
# LIGHTGBM CONFIGURATION AND HYPER-PARAMETERS
GENERATE_SUBMISSION_FILES = True
EVALUATE_VALIDATION_SET = True
STRATIFIED_KFOLD = False
RANDOM_SEED = 737851
NUM_FOLDS = 10
EARLY_STOPPING = 100
ROOT            = Path("./")

LIGHTGBM_PARAMS = {
    'boosting_type': 'goss',
    'n_estimators': 10000,
    'learning_rate': 0.005134,
    'num_leaves': 54,
    'max_depth': 10,
    'subsample_for_bin': 240000,
    'reg_alpha': 0.436193,
    'reg_lambda': 0.479169,
    'colsample_bytree': 0.508716,
    'min_split_gain': 0.024766,
    'subsample': 1,
    'is_unbalance': False,
    'silent':-1,
    'verbose':-1,
    
}


# ### Set aggregations

# In[5]:


# AGGREGATIONS
APPLPREV1_AGG = {
    'annuity_853A' : ['min', 'max', 'mean'],
    'currdebt_94A' : ['max', 'mean', 'sum'] ,
    'mainoccupationinc_437A' : ['max', 'mean', 'sum'] ,
    'cancelreason_3545846M' : ['mean']
}
APPLPREV2_AGG = {
    
}
PERSON1_AGG={}
PERSON2_AGG={}
OTHER_AGG={}
DEBITCARD_AGG={}
TAX_REGISTRY_A_AGG={}
TAX_REGISTRY_B_AGG={}
TAX_REGISTRY_C_AGG={}
CREDIT_BUREAU_B_1_AGG={}
CREDIT_BUREAU_B_2_AGG={}


# # **MAIN FUNCTION**

# In[6]:


def main(debug= False):
    num_rows = 1111 if debug else None
    with timer("base"):
        df = get_base(DATA_DIRECTORY, num_rows=num_rows)
        print("base dataframe shape:", df.shape)

    with timer("static"):
        df_static = get_static(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_static, on='case_id', how='left', suffix='_static')
        print("static dataframe shape:", df_static.shape)
        del df_static
        gc.collect()

    with timer("static_cb"):
        df_static_cb = get_static_cb(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_static_cb, on='case_id', how='left', suffix='_static_cb')
        print("static cb dataframe shape:", df_static_cb.shape)
        del df_static_cb
        gc.collect()

    with timer("Previous applications depth 1 test"):
        df_applprev1 = get_applprev1(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_applprev1, on='case_id', how='left', suffix='_applprev1')
        print("Previous applications depth 1 test dataframe shape:", df_applprev1.shape)
        del df_applprev1
        gc.collect()

    with timer("Previous applications depth 2 test"):
        df_applprev2 = get_applprev2(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_applprev2, on='case_id', how='left', suffix='_applprev2')
        print("Previous applications depth 2 test dataframe shape:", df_applprev2.shape)
        del df_applprev2
        gc.collect()

    with timer("Person depth 1 test"):
        df_person1 = get_person1(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_person1, on='case_id', how='left', suffix='_person1')
        print("Person depth 1 test dataframe shape:", df_person1.shape)
        del df_person1
        gc.collect()

    with timer("Person depth 2 test"):
        df_person2 = get_person2(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_person2, on='case_id', how='left', suffix='_person2')
        print("Person depth 2 test dataframe shape:", df_person2.shape)
        del df_person2
        gc.collect()

    with timer("Other test"):
        df_other = get_other(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_other, on='case_id', how='left', suffix='_other')
        print("Other test dataframe shape:", df_other.shape)
        del df_other
        gc.collect()

    with timer("Debit card test"):
        df_debitcard = get_debitcard(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_debitcard, on='case_id', how='left', suffix='_debitcard')
        print("Debit card test dataframe shape:", df_debitcard.shape)
        del df_debitcard
        gc.collect()

    with timer("Tax registry a test"):
        df_tax_registry_a = get_tax_registry_a(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_tax_registry_a, on='case_id', how='left', suffix='_tax_registry_a')
        print("Tax registry a test dataframe shape:", df_tax_registry_a.shape)
        del df_tax_registry_a
        gc.collect()

    with timer("Tax registry b test"):
        df_tax_registry_b = get_tax_registry_b(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_tax_registry_b, on='case_id', how='left', suffix='_tax_registry_b')
        print("Tax registry b test dataframe shape:", df_tax_registry_b.shape)
        del df_tax_registry_b
        gc.collect()

    with timer("Tax registry c test"):
        df_tax_registry_c = get_tax_registry_c(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_tax_registry_c, on='case_id', how='left', suffix='_tax_registry_c')
        print("Tax registry c test dataframe shape:", df_tax_registry_c.shape)
        del df_tax_registry_c
        gc.collect()
    '''
    with timer("Credit bureau a 1 test"):
        df_credit_bureau_a_1 = get_credit_bureau_a_1(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_credit_bureau_a_1, on='case_id', how='left', suffix='_cb_a_1')
        print("Credit bureau a 1 test dataframe shape:", df_credit_bureau_a_1.shape)
        del df_credit_bureau_a_1
        gc.collect()
        '''

    with timer("Credit bureau b 1 test"):
        df_credit_bureau_b_1 = get_credit_bureau_b_1(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_credit_bureau_b_1, on='case_id', how='left', suffix='_cb_b_1')
        print("Credit bureau b 1 test dataframe shape:", df_credit_bureau_b_1.shape)
        del df_credit_bureau_b_1
        gc.collect()

    '''
    with timer("Credit bureau a 2 test"):
        df_credit_bureau_a_2 = get_credit_bureau_a_2(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_credit_bureau_a_2, on='case_id', how='left', suffix='_cb_a_2')
        print("Credit bureau a 2 test dataframe shape:", df_credit_bureau_a_2.shape)
        # Free memory
        del df_credit_bureau_a_2
        gc.collect()
'''   
    with timer("Credit bureau b 2 test"):
        df_credit_bureau_b_2 = get_credit_bureau_b_2(DATA_DIRECTORY, num_rows=num_rows)
        df = df.join(df_credit_bureau_b_2, on='case_id', how='left', suffix='_cb_b_2')

    
    with timer("Feature engineering / preprocessing"):    
        df=feature_engineering(df)
   
   
    with timer("Model training"):
        df, cat_cols = to_pandas(df)
        model = kfold_lightgbm_sklearn(df, cat_cols)
       
    with timer("Feature importance assesment"):
        get_features_importances(df, model)
        
    with timer("Submission"):
        if generate_submission_file(df, model):
            "Submission file has been created."
        
    del df
    del model
    
    print("NOTEBOOK HAS BEEN SUCCESSFULLY EXECUTED !!!")


# # **UTILITY FUNCTIONS**

# ### Pipeline

# In[7]:


class Pipeline:
    @staticmethod
    
    
    # Sets datatypes accordingly
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))            

        return df
    
    
    # Changes the values of all date columns. The result will not be a date but number of days since date_decision.
    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                
        df = df.drop("date_decision", "MONTH")

        return df
    
    # It drops columns with a lot of NaN values.
    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


# In[8]:


def get_info(dataframe):
    """
    View data types, shape, and calculate the percentage of NaN (missing) values in each column
    of a Polars DataFrame simultaneously.
    
    Parameters:
    dataframe (polars.DataFrame): The DataFrame to analyze.
    
    Returns:
    None
    """
    # Print DataFrame shape
    print("DataFrame Shape:", dataframe.shape)
    print("-" * 60)
    
    # Print column information
    print("{:<50} {:<30} {:<20}".format("Column Name", "Data Type", "NaN Percentage"))
    print("-" * 60)
    
    # Total number of rows in the DataFrame
    total_rows = len(dataframe)
    
    # Iterate over each column
    for column in dataframe.columns:
        # Get the data type of the column
        dtype = str(dataframe[column].dtype)
        
        # Count the number of NaN values in the column
        nan_count = dataframe[column].null_count()
        
        # Calculate the percentage of NaN values
        nan_percentage = (nan_count / total_rows) * 100
        
        # Print the information
        print("{:<50} {:<30} {:.2f}%".format(column, dtype, nan_percentage))


# In[9]:


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data, cat_cols


# In[10]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[11]:


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))


# In[12]:


def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    

    temp=base.loc[:, ["WEEK_NUM", "target", "score"]] \
        .sort_values("WEEK_NUM") \
        .groupby("WEEK_NUM").mean()
   
    week_nums_to_drop = temp[(temp["target"] == 0) | (temp["target"] == 1)].index.tolist()

    base_filtered = base[~base["WEEK_NUM"].isin(week_nums_to_drop)]

    # Apply the aggregator
    gini_in_time = base_filtered.loc[:, ["WEEK_NUM", "target", "score"]] \
        .sort_values("WEEK_NUM") \
        .groupby("WEEK_NUM")[["target", "score"]] \
        .apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()

    

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.nanmean(gini_in_time)  # Use np.nanmean to handle NaN values
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


# #  **MODEL**

# In[13]:


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
    
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        # Use tqdm to create a progress bar during the prediction
        with tqdm(total=len(self.estimators), desc="Predicting", unit=" models") as pbar:
            for i, estimator in enumerate(self.estimators):
                y_preds[i] = estimator.predict_proba(X)
                pbar.update(1)  # Update the progress bar
        return np.mean(y_preds, axis=0)

    
    def get_splits(self, aggregation_method=np.mean):
        
        feature_importances_list=[]
        for x in self.estimators:
            feature_importances_list.append(x.booster_.feature_importance(importance_type='split'))
            
        # Aggregate feature importances across all models
        if all(importances is not None for importances in feature_importances_list):
            combined_importances = aggregation_method(feature_importances_list, axis=0)
        else:
            combined_importances = None   
        return combined_importances
    
    
    def get_gains(self, aggregation_method=np.mean):
        
        feature_importances_list=[]
        for model in self.estimators:
            feature_importances_list.append(x.booster_.feature_importance(importance_type='gain'))
            
        # Aggregate feature importances across all models
        if all(importances is not None for importances in feature_importances_list):
            combined_importances = aggregation_method(feature_importances_list, axis=0)
        else:
            combined_importances = None
              
        return combined_importances
    
    def get_features_importances_df(self, df):
        del_features = ['target', 'case_id']
        predictors = list(filter(lambda v: v not in del_features, df.columns))
        importance_df = pd.DataFrame()
        eval_results = dict()
        for model in self.estimators:
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = predictors
            fold_importance["gain"] = model.booster_.feature_importance(importance_type='gain')
            fold_importance["split"] = model.booster_.feature_importance(importance_type='split')
            importance_df = pd.concat([importance_df, fold_importance], axis=0)
        return importance_df


# In[14]:


def kfold_lightgbm_sklearn(data, categorical_feature = None):
    start_time = time.time()
    df = data[data['target'].notnull()]
    test = data[data['target'].isnull()]
    print("Train/valid shape: {}, test shape: {}".format(df.shape, test.shape))
    del_features = ['target', 'case_id']
    predictors = list(filter(lambda v: v not in del_features, df.columns))

    if not STRATIFIED_KFOLD:
        folds = KFold(n_splits= NUM_FOLDS, shuffle=True, random_state= RANDOM_SEED)
    else:
        folds = StratifiedKFold(n_splits= NUM_FOLDS, shuffle=True, random_state= RANDOM_SEED)
    
        # Hold oof predictions, test predictions, feature importance and training/valid auc
    oof_preds = np.zeros(df.shape[0])
    
    importance_df = pd.DataFrame()
    eval_results = dict()
    
    fitted_models = []
    with tqdm(total=NUM_FOLDS) as pbar:
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[predictors], df['target'])):
            train_x, train_y = df[predictors].iloc[train_idx], df['target'].iloc[train_idx]
            valid_x, valid_y = df[predictors].iloc[valid_idx], df['target'].iloc[valid_idx]

            params = {'random_state': RANDOM_SEED, 'nthread': NUM_THREADS}
            clf = LGBMClassifier(**{**params, **LIGHTGBM_PARAMS})


            if not categorical_feature:
                    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                            eval_metric='auc' )
            else:
                clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],eval_metric='auc',
                        feature_name= list(df[predictors].columns), categorical_feature= categorical_feature)


            fitted_models.append(clf)

            if EVALUATE_VALIDATION_SET:
                oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]



                # Feature importance by GAIN and SPLIT

            eval_results['train_{}'.format(n_fold+1)]  = clf.evals_result_['training']['auc']
            eval_results['valid_{}'.format(n_fold+1)] = clf.evals_result_['valid_1']['auc']

            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (NUM_FOLDS - n_fold - 1) / (n_fold + 1)
            print('Fold %2d AUC : %.6f. Elapsed time: %.2f seconds. Remaining time: %.2f seconds.'
                  % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]), elapsed_time, remaining_time))
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()
            pbar.update(1)
            
    print('Full AUC score %.6f' % roc_auc_score(df['target'], oof_preds))
    # Get the average feature importance between folds
    
    
    
    if len(df)>0:
        base=get_base(DATA_DIRECTORY, len(df))
        base, cat_cols = to_pandas(base)
        base=base[base['target'].notnull()]
        base['score']= oof_preds
        gini_score = gini_stability(base)
        print("Gini Score of the valid set:", gini_score)
    
    
    
    
    # Save feature importance, test predictions and oof predictions as csv
    if GENERATE_SUBMISSION_FILES:

        # Generate oof csv
        oof = pd.DataFrame()
        oof['case_id'] = df['case_id'].copy()
        df['PREDICTIONS'] = oof_preds.copy()
        df['target'] = df['target'].copy()
        df.to_csv('oof{}.csv'.format(SUBMISSION_SUFIX), index=False)
        
        
        
    model = VotingModel(fitted_models)
    return model


# # **SUBMISSION**

# In[15]:


def generate_submission_file(data, model):
    
    test = data[data['target'].isnull()]
    del_features = ['target', 'case_id']
    predictors = list(filter(lambda v: v not in del_features, data.columns))
    y_pred = pd.Series(model.predict_proba(test[predictors])[:, 1], index=test[predictors].index)    
    df_subm = pd.read_csv(ROOT / "sample_submission.csv")
    df_subm = df_subm.set_index("case_id")
    df_subm["score"] = y_pred
    df_subm.to_csv("submission.csv")
    
    return True
    
    


# # **EVALUATE FEATURES IMPORTANCES**

# In[16]:


def get_features_importances(data, model):
    importance_df = model.get_features_importances_df(data)
    mean_importance = importance_df.groupby('feature').mean().reset_index()
    mean_importance.sort_values(by= 'gain', ascending=False, inplace=True)
    mean_importance.to_csv('feature_importance{}.csv'.format(SUBMISSION_SUFIX), index=False)
    return True


# # **FEATURE ENGINEERING FUNCTION**

# In[17]:


def feature_engineering(df):
    with Pool(NUM_THREADS) as p:
        df = p.map(Pipeline.handle_dates, [df]*NUM_THREADS)
        df = p.map(Pipeline.filter_cols, [df]*NUM_THREADS)
    return df


# # **GET FUNCTIONS**

# In[18]:


def group(df_to_agg, prefix, aggregations, aggregate_by='case_id'):
    # Create a dictionary mapping aggregation functions to their string representations
    func_mapping = {
        'min': pl.min,
        'max': pl.max,
        'mean': pl.mean,
        'sum': pl.sum
    }
    
# Perform the aggregation
    agg_df = df_to_agg.group_by(aggregate_by).agg(**{
        f"{func}_{col}": func_mapping[func](col) for col, funcs in aggregations.items() for func in funcs
    })
    '''
    # Rename columns
    for col, funcs in aggregations.items():
        for func in funcs:
            old_name = f"{col}_{func}"
            new_name = f"{prefix}{col}_{func.upper()}"
            agg_df = agg_df.select(pl.col(old_name).alias(new_name))
    '''
    return agg_df


# ### get_base()

# In[19]:


def get_base(path, num_rows = None):
    # Read the Parquet file using scan() method
    train={}
    test={}
    
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_base.parquet'))
        
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_base.parquet')).limit(num_rows) 
        
    test = pl.read_parquet(os.path.join(path, 'test/test_base.parquet'))    
    length=len(test)
    nan_series=pl.Series([None] * length)
    test = test.select(pl.col("*"), nan_series.alias("target"))
    df=pl.concat([train, test])
    df = df.with_columns(pl.col('date_decision').cast(pl.Date))
    return df


# In[20]:


A=get_base(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_static()

# In[21]:


def get_static(path, num_rows = None):
# Read the Parquet file using scan() method
    chunks = []
    for path in glob(DATA_DIRECTORY+str('train/train_static_0_*.parquet')):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
    train = (pl.concat(chunks, how="vertical_relaxed")).pipe(Pipeline.filter_cols)
    
    if num_rows!= None:
        df1 = train.slice(0,num_rows)
        df2 = train.slice(num_rows,len(train))
        
        train=df1
        del df2
    
    chunks = []
    for path in glob(DATA_DIRECTORY+str('test/test_static_0_*.parquet')):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
        test = pl.concat(chunks, how="vertical_relaxed")
    
    
    columns_to_keep = train.columns

# Find columns in 'test' that are not in 'train'
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]

# Drop columns from 'test' that are not in 'train'
    test = test.drop(columns_to_remove)
    df=pl.concat([train, test])
    return df


# In[20]:


A=get_static(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_static_cb()

# In[22]:


def get_static_cb(path, num_rows = None):
    
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_static_cb_0.parquet')).pipe(Pipeline.set_table_dtypes)
        
        
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_static_cb_0.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
       
    
    test = pl.read_parquet(os.path.join(path, 'test/test_static_cb_0.parquet')).pipe(Pipeline.set_table_dtypes)
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    return df


# In[23]:


A=get_static_cb(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_applprev1(DATA_DIRECTORY, num_rows=num_rows)

# In[24]:


def get_applprev1(path, num_rows = None):
    
    
    chunks = []
    for path in glob(DATA_DIRECTORY+str('train/train_applprev_1_*.parquet')):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
    train = pl.concat(chunks, how="vertical_relaxed").pipe(Pipeline.filter_cols)
    
    
    if num_rows!= None:
        df1 = train.slice(0,num_rows)
        df2 = train.slice(num_rows,len(train))

        train=df1
        del df2   
    
    chunks = []
    for path in glob(DATA_DIRECTORY+str('test/test_applprev_1_*.parquet')):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
        test = pl.concat(chunks, how="vertical_relaxed")
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)
    df=pl.concat([train, test])
    agg_df = group(df, '', APPLPREV1_AGG)
    del df 
    return agg_df


# In[25]:


A=get_applprev1(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_applprev2(DATA_DIRECTORY, num_rows=num_rows)

# In[26]:


def get_applprev2(path, num_rows = None):
    train={}
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_applprev_2.parquet')).pipe(Pipeline.set_table_dtypes)
        
     
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_applprev_2.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
       
    
    
    test = pl.read_parquet(os.path.join(path, 'test/test_applprev_2.parquet')).pipe(Pipeline.set_table_dtypes)
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', APPLPREV2_AGG)
    del df 
    return agg_df


# In[26]:


A=get_applprev2(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_person1

# In[27]:


def get_person1(path, num_rows = None):
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_person_1.parquet')).pipe(Pipeline.set_table_dtypes)
        
    
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_person_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
      
    
    
    test = pl.read_parquet(os.path.join(path, 'test/test_person_1.parquet')).pipe(Pipeline.set_table_dtypes)
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', PERSON1_AGG)
    del df
    
    return agg_df


# In[28]:


A=get_person1(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_person2

# In[28]:


def get_person2(path, num_rows = None):
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_person_2.parquet')).pipe(Pipeline.set_table_dtypes)
        
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_person_2.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
        
    test = pl.read_parquet(os.path.join(path, 'test/test_person_2.parquet')).pipe(Pipeline.set_table_dtypes)
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', PERSON2_AGG)
    del df
    
    return agg_df


# In[30]:


A=get_person2(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### other

# In[29]:


def get_other(path, num_rows = None):
     # Read the Parquet file using scan() method
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_other_1.parquet')).pipe(Pipeline.set_table_dtypes)
        
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_other_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
         
    test = pl.read_parquet(os.path.join(path, 'test/test_other_1.parquet')).pipe(Pipeline.set_table_dtypes)
    
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', OTHER_AGG)
    del df
    
    return agg_df


# In[32]:


A=get_other(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ## get_debitcard

# In[30]:


def get_debitcard(path, num_rows = None):
    # Read the Parquet file using scan() method
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_debitcard_1.parquet')).pipe(Pipeline.set_table_dtypes)
        
     
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_debitcard_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
      
        
    test = pl.read_parquet(os.path.join(path, 'test/test_debitcard_1.parquet')).pipe(Pipeline.set_table_dtypes)
    
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', DEBITCARD_AGG)
    del df
    
    return agg_df


# In[34]:


A=get_debitcard(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_tax_registry_a

# In[31]:


def get_tax_registry_a(path, num_rows = None):
    
    # Read the Parquet file using scan() method
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_tax_registry_a_1.parquet')).pipe(Pipeline.set_table_dtypes)
        
    
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_tax_registry_a_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
  
    
    
    test = pl.read_parquet(os.path.join(path, 'test/test_tax_registry_a_1.parquet')).pipe(Pipeline.set_table_dtypes)
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', TAX_REGISTRY_A_AGG)    
    del df
    
    return agg_df


# In[36]:


A=get_tax_registry_a(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_tax_registry_b

# In[32]:


def get_tax_registry_b(path, num_rows = None):
    # Read the Parquet file using scan() method
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_tax_registry_b_1.parquet')).pipe(Pipeline.set_table_dtypes)
        
        
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_tax_registry_b_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
        
    
    test = pl.read_parquet(os.path.join(path, 'test/test_tax_registry_b_1.parquet')).pipe(Pipeline.set_table_dtypes)
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', TAX_REGISTRY_B_AGG) 
    del df
    
    return agg_df


# In[38]:


A=get_tax_registry_b(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_tax_registry_c

# In[33]:


def get_tax_registry_c(path, num_rows = None):
     # Read the Parquet file using scan() method
# Read the Parquet file using scan() method
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_tax_registry_c_1.parquet')).pipe(Pipeline.set_table_dtypes)
    
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_tax_registry_c_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
        
    
    test = pl.read_parquet(os.path.join(path, 'test/test_tax_registry_c_1.parquet')).pipe(Pipeline.set_table_dtypes)
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', TAX_REGISTRY_C_AGG)    
    del df
    
    return agg_df


# In[40]:


A=get_tax_registry_c(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_credit_bureau_a_1

# In[34]:


def get_credit_bureau_a_1(path, num_rows = None):
    chunks = []
    for path in glob(DATA_DIRECTORY+str('train/train_credit_bureau_a_1_*.parquet')):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
    train = pl.concat(chunks, how="vertical_relaxed").pipe(Pipeline.filter_cols)
    if num_rows!= None:
        df1 = train.slice(0,num_rows)
        df2 = train.slice(num_rows,len(train))
        
        train=df1
        del df2
    
    
    chunks = []
    for path in glob(DATA_DIRECTORY+str('test/test_credit_bureau_a_1_*.parquet')):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
        test = pl.concat(chunks, how="vertical_relaxed")
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)
    df=pl.concat([train, test])
    agg_df = group(df, '', CREDIT_BUREAU_A_1_AGG) 
    del df
    
    return agg_df


# ### get_credit_bureau_b_1

# In[35]:


def get_credit_bureau_b_1(path, num_rows = None):
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_credit_bureau_b_1.parquet')).pipe(Pipeline.set_table_dtypes)
        
        
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_credit_bureau_b_1.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)
   
    
    test = pl.read_parquet(os.path.join(path, 'test/test_credit_bureau_b_1.parquet')).pipe(Pipeline.set_table_dtypes)
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)

    df=pl.concat([train, test])
    agg_df = group(df, '', CREDIT_BUREAU_B_1_AGG) 
    
    
    del df
    
    return agg_df


# In[43]:


A=get_credit_bureau_b_1(DATA_DIRECTORY, 10000)
get_info(A)
del A


# ### get_credit_bureau_a_2

# In[36]:


def get_credit_bureau_a_2(path, num_rows = None):
    chunks = []
    for path in glob(DATA_DIRECTORY+str('train/train_credit_bureau_a_2_*.parquet')):
        chunks.append(reduce_mem_usage(pl.read_parquet(path))) #.pipe(Pipeline.set_table_dtypes))
        print(path)
    train = pl.concat(chunks, how="vertical_relaxed").pipe(Pipeline.filter_cols)
    
    '''
    if num_rows!= None:
        df1 = train.slice(0,num_rows)
        df2 = train.slice(num_rows,len(df))
        
        train=df1
        del df2
    
    '''
    chunks = []
    for path in glob(DATA_DIRECTORY+str('test/test_credit_bureau_a_2_*.parquet')):
        chunks.append(reduce_mem_usage(pl.read_parquet(path))) #.pipe(Pipeline.set_table_dtypes))
        test = pl.concat(chunks, how="vertical_relaxed")
        print(path)
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)
    df=pl.concat([train, test])
    return df
                      
                      
   


# ### get_credit_bureau_b_2

# In[37]:


def get_credit_bureau_b_2(path, num_rows = None):
    if num_rows == None:
        train = pl.read_parquet(os.path.join(path, 'train/train_credit_bureau_b_2.parquet')).pipe(Pipeline.set_table_dtypes)
   
    else:
        train = pl.read_parquet(os.path.join(path, 'train/train_credit_bureau_b_2.parquet')).limit(num_rows).pipe(Pipeline.set_table_dtypes)

    
    test = pl.read_parquet(os.path.join(path, 'test/test_credit_bureau_b_2.parquet')).pipe(Pipeline.set_table_dtypes)
    
    train = train.pipe(Pipeline.filter_cols)
   
    columns_to_keep = train.columns
    columns_to_remove = [column for column in test.columns if column not in columns_to_keep]
    test = test.drop(columns_to_remove)
    
    df=pl.concat([train, test])
    agg_df = group(df, '', CREDIT_BUREAU_B_2_AGG) 
    
    del df
    
    return agg_df


# In[46]:


A=get_credit_bureau_b_2(DATA_DIRECTORY, 10000)
get_info(A)
del A


# # **EXECUTION**

# In[40]:


if __name__ == "__main__":
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.max_columns', 100)
    with timer("Pipeline total time"):
        main(debug= True)
