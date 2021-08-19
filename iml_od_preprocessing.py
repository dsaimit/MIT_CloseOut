import pandas as pd
import numpy as np

def phase_mapping(x):
    """
    
    """
    
    phase_dict = {'I':['PHASE 1', 'PHASE I','I', 'IB', 'I/II'], 'II':['PHASE 2', 'II', 'IIA', 'IIB', 'II/III','PHASE IIB','PHASE IIA','PHASE II'],
                  'III': ['III', 'IIIB','PHASE III','PHASE IIIA','IIIA'], 'IV': ['PHASE 4']}
        
    if x in phase_dict['I']:
        return 'I'
    
    elif x in phase_dict['II']:
        return 'II'
    
    elif x in phase_dict['III']:
        return 'III' 
    
    elif x in phase_dict['IV']:
        return 'IV'
    
    else:
        return x

def ta_mapping(x):
    """
    
    """
    
    ta_dict = {'TA_Other': ['CVM', 'Plasma Derived Therapy', 'Vaccine', 'Ophthalmology', 'Others']}
    
    if x in ta_dict['TA_Other']:
        return 'TA_Other'
    else:
        return x
    
def one_hot_encoding_one_column(df, col):
    
    one_hot_cols = pd.get_dummies(df[col])
    
    df = df.drop([col],axis = 1)
    # Join the encoded df
    df = df.join(one_hot_cols)
    
    return df
    

def CRO_preprocessing(df):
    
    """
    
    Let's change CRO names to most widespread and others
    
    input: df - initial dataframe with not processed column CRO
    
    output: df - changed dataset with one-hot-encoding of most widespread CROs or replacement with "Other"
    
    """ 
    
    assert 'CRO' in df.columns

    df.loc[:, 'CRO'] = df.CRO.str.replace(
        'Quintiles', 'IQVIA') #To reflect Quintiles merger with IQVIA
    df.loc[:, 'CRO'] = df.CRO.str.replace(
        'PRA Health Sciences', 'PRA')
    df.loc[:, 'CRO'] = df.CRO.str.replace(
        'Takeda PRA Development Center', 'PRA') #To reflect PRA M&A of Takeda JV
    df.loc[:, 'CRO'] = df.CRO.str.replace(
        'MEDISCIENCE PLANNING INC', 'MEDISCIENCE')

    # Introducing new column to refer to "Other" CROs 
    CRO_list = ['IQVIA', 'PPD', 'PRA', 'Celerion']
    df['GeneralCRO'] = df['CRO']
    conditions_CRO = ~df['CRO'].isin(CRO_list)

    df.loc[conditions_CRO, ['GeneralCRO']] = 'Other'

    one_hot_cro = pd.get_dummies(df['GeneralCRO'])

    df = df.drop(['CRO','GeneralCRO'],axis = 1)
    # Join the encoded df
    df = df.join(one_hot_cro)
   
    return df

def country_preprocessing(df):
    
    """
    
    Let's use as binary columns whether a trial was in the US, 
    the UK, Canada, Spain, Italy, Japan 
    as well as number of countries in clinical trial
    
    input: df - initial dataframe with not processed column CRO
    
    output: df - changed dataset with one-hot-encoding of most 
    widespread CROs or replacement with "Other"
    
    
    """
    
    # lets start from counting number of countries
    
    num_countries = []
    
    for c in df.COUNTRY.to_list():
        try:
            c = c.split(',')
            num_countries.append(len(c))
        except:
            num_countries.append(np.nan)
    
    df['Num_Countries'] = num_countries
    
    # Now lets create binary variables whether particular countries were involved in a clinical trial
    
    target_countries = {'US': ' United States of America (the)',
                        'UK': ' United Kingdom of Great Britain and Northern Ireland (the)',
                        'CANADA': ' Canada',
                        'SPAIN': ' Spain',
                        'ITALY': ' Italy',
                        'JAPAN': ' Japan'}
    
    def country_occurance_bin(countries_list, target_country):
    
        bin_col_country = []

        for c in countries_list:
            try:
                if target_country in c:
                    bin_col_country.append(True)
                else:
                    bin_col_country.append(False)
            except:
                bin_col_country.append(False)

        return bin_col_country
    
    for key, val in target_countries.items():
        df[key] = country_occurance_bin(df['COUNTRY'].values, val)
        df[key] = df[key]*1
        
    df = df.drop(['COUNTRY'], axis = 1)
    
    return df
    
    
    
    
    
    

