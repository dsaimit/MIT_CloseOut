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

1. **CAR_Raw_Data.csv** - Raw CAR data (2013-2021 years) from the internal Takeda Dashboard
2. **classification_results.csv** - Classification In-sample and Out-of-sample quality metrics for all classification models after the validation stage.
3. **close_out_preprocessed_data_20210908.csv** - CAR data where we processed that close out takes into account unlocks and relocks time
4. **DM_RT_DATA_INGEST_AUTOMATN_LOGS_PROCESSING.csv** - 
5. **expected_lso_dbl.csv** - 
6. **expected_lso_dbl_trial_id_matching.csv** - 
7. **lso_to_dbl.csv** - 
8. **max_dates_unlocks.csv** - 
9. **ml_df.csv** - Processed CAR dataset that is ready for ML analysis
10. **post_plots_CAR.csv** - modified processed CAR data so that we can use the data for outlier analysis.
11. **Probability_Success.csv** - 
12. **processed_CAR_data.csv** - 
13. **processed_DBUL_data.csv** - Merged tables of CAR and Database Unlocks
14. **regression_results.csv** - 
15. **sql_table_descriptions.csv** - 
16. **trial_ids.csv** - 
17. **trial_ids_car_data.csv** - 
18. **unlocks_data.csv** - Raw data of unlocks for 2019-2021 from Dan Godoy. Columns B and C were created by us to match trial ids between this data and CAR.

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

1. **close_classification_analysis_v1.ipynb** - file with classification analysis of predicting whether DBL_Actual is going to be before or coincide with Expected_DBL. 
3. **closeout_regression_analysis_v3.ipynb** - 







