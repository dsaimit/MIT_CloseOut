# MIT Capstone Project 2021 - Understanding what causes suboptimal operational performancein clinical trials

## Authors: Denis Sai & Edoardo Italia

## Project Sponsor: Saurabh Awasthi

## MIT Advisor: Prof. Steve Graves

# Documentation

## MIT Final Deliverables

We had 4 final deliverables (3 of them are in _MIT_Deliverables_ folder):

1. Final Project Report (for private usage) - _Takeda_Denis_Sai_Edoardo_Italia_Final_Report.pdf_
2. Project Poster (publicly available) -  _Takeda_Denis_Sai_Edoardo_Italia_Poster.pdf_
3. Project Presentation (publicly available) - _Takeda_Denis_Sai_Edoardo_Italia_Presentation.pdf_
4. Project Video (publicly available) - due t a file size, we add it as a link - https://drive.google.com/file/d/1m5CIPkrLijcl3-YoTzSQ_BBf0nD9P_o1/view

## Dependencies

Library Dependencies are specified in _dependencies.txt_

Python version 3.8.8

## Data

Here is the description of all raw and processed CSV files we used (in alphabetical order):

1. **CAR_Raw_Data.csv** - Raw CAR data (2013-2021 years) from the internal Takeda Dashboard (this data comes directly from CAR)
2. **classification_results.csv** - Classification In-sample and Out-of-sample quality metrics for all classification models after the validation stage.
3. **close_out_preprocessed_data_20210908.csv** - CAR data where we processed that close out takes into account unlocks and relocks time
4. **DM_RT_DATA_INGEST_AUTOMATN_LOGS_PROCESSING.xlsx** - This is an extract of results of the data quality check that we perform when we ingest clinical trial data into the system. Got it from Niranjan Saigaonkar.
5. **expected_lso_dbl.csv** - from MSPS we found for each clinical trial first snapshot information with available LSO and DBL and used it as expected values for LSO and DBL- from this we also computed expected close-out time.
6. **expected_lso_dbl_trial_id_matching.csv** - matching between MSPS and CAR of trial_ids.
7. **max_dates_unlocks.csv** - for each trial with unlocks, find maximum date of relock if available.
8. **ml_df.csv** - Processed CAR dataset that is ready for ML analysis.
9. **post_plots_CAR.csv** - modified processed CAR data so that we can use the data for outlier analysis.
10. **Probability_Success.csv** - external information about compounded probabilities of going-to-market from each stage of clinical trial depending on phase and TA
11. **processed_CAR_data.csv** - data wrangling over CAR_Raw_Data.csv, mostly around CRO, TA and Phase handling as well as creation of all cycle times before Close-Out stage
12. **processed_DBUL_data.csv** - Merged tables of CAR and Database Unlocks.
13. **regression_results.csv** - Regression In-sample and Out-of-sample quality metrics for all regression models after the validation stage.
14. **sql_table_descriptions.csv** - manual descriptions of all datasets from Datahub when we tried to understand initially what data is usefull for Close-Out analysis
15. **trial_ids_car_data.csv** - matching of trial_ids between CAR data and DM_RT_DATA_INGEST_AUTOMATN_LOGS_PROCESSING.
16. **unlocks_data.csv** - Raw data of unlocks for 2019-2021 from Dan Godoy. Columns B and C were created by us to match trial ids between this data and CAR.

## Py files

1. **binary_classification.py** - file with all classification validation procedures and functions for final quality metrics calculations (In-sample and Out-of-sample ROC-AUC, F1 Score).
2. **iml_od_modelling.py** - file with al regression validation procedures and functions for final quality metrics calculations (In-sample and Out-of-sample MSE, MAE, R^2).
3. **imd_od_preprocessing.py** - file with all data preprocessing for CAR data (categorical variables handling, CRO data grouping, etc)


## Exploratory Data Analysis

Set of files that create EDA analysis describing Close Out stage and Unlocks analysis

_Close-Out EDA analysis:_

1. **CAR_boxplot_analysis.ipynb** - We create scatterplots of close out performance per CRO, TA and Phase with group medians. Also here we performed analysis of comparing distributions before and after COVID to find whether we need to account for COVID in our set of independent variables. _Py-file dependencies: none. Data dependencies: close_out_preprocessed_data_20210908.csv_
2. **CAR_Outlier_Analysis.ipynb** - Create distirbutions of quartiles across different groups. _Py-file dependencies: none. Data dependencies: post_plots_CAR.csv_

_Database Unlocks Analysis:_

1. **DBUL_Plots.ipynb** - Count plots for Database Unlocks for 2019-2021. _Py-file dependencies: none. Data dependencies: unlocks_data.csv_
2. **DBUL_Visualizations.ipynb** - Prevalence plots, distribution plots for Database Unlocks for 2019-2021. _Py-file dependencies: none. Data dependencies: processed_DBUL_data.csv_


## Time-Series Analysis of Close Out

1. **CAR_Seasonality_Causal_Impact.ipynb** - Decompose Close-out time series into trend and seasonality. Also it applied Causal_Impact library to identify impact from COVID. _Py-file dependencies: none. Data dependencies: ml_df.csv_
2. **CAR_Trend_Plots.ipynb** - Notebook with some statistical tests that check whether our Close-Out time series has an intrinsic trend (Dickey–Fuller test). _Py-file dependencies: none. Data dependencies: ml_df_

## Machine Learning Analysis

**close_classification_analysis_v1.ipynb** - file with classification analysis of predicting whether DBL_Actual is going to be before or coincide with Expected_DBL. 

Models that we use here: 
1. L1 Logistic Regression
2. L2 Logistic Regression
3. Decision Trees
4. Random Forest
5. Gradient Boosting
6. Categorical Boosting

_Py-file dependencies: binary_classification.py, imd_od_preprocessing.py_

_Data dependincies: processed_CAR_data.csv, trial_ids_car_data.csv, DM_RT_DATA_INGEST_AUTOMATN_LOGS_PROCESSING.xlsx, max_dates_unlocks.csv, expected_lso_dbl.csv, expected_lso_dbl_trial_id_matching.csv_

_Results file: classification_results.csv_

**closeout_regression_analysis_v3.ipynb** - file with regression analysis of close-out time.

Models that we use here:
1. Vanilla Linear Regression
2. L1 Linear Regression
3. L2 Linear Regression
4. Poisson Regression
5. Decision Trees
6. Gradient Boosting

_Py-file dependencies: binary_classification.py, imd_od_preprocessing.py_

_Data dependincies: processed_CAR_data.csv, trial_ids_car_data.csv, DM_RT_DATA_INGEST_AUTOMATN_LOGS_PROCESSING.xlsx, max_dates_unlocks.csv, expected_lso_dbl.csv, expected_lso_dbl_trial_id_matching.csv, Probability_Success.csv_

_Results file: regression_results.csv_






