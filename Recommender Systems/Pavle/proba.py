# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:11:12 2016

@author: Pavle
"""

import pandas as pd
import numpy as np


def get_disj_tags():
    """Finds all tags and return disjunctive list"""
    
    data=pd.read_csv('./Data/item_profile.csv', sep='\t', header=0, usecols=['tags'])
    
    all_tags = data.loc[:]['tags'].values
    
    disj = set()
    
    for i in all_tags:
        tmp = set(str(i).split(','))
        disj = disj.union(tmp)

    return np.array(list(disj))

def get_disj_titles():
    """Finds all tags and return disjunctive list"""
    
    data=pd.read_csv('./Data/item_profile.csv', sep='\t', header=0, usecols=['title'])
    
    all_tags = data.loc[:]['title'].values
    
    disj = set()
    
    for i in all_tags:
        tmp = set(str(i).split(','))
        disj = disj.union(tmp)

    return list(disj)
    
#print(get_disj_titles())

def get_max_num_of_list_elems_in_col(colname):
    
    """
        Finds max num of elements in list which represent column element
        
        Ex: get_max_num_of_list_elems_in_col("tags")  # 57
            get_max_num_of_list_elems_in_col("title") #11
    """
    
    maxN = 0
    
    for i in data[[colname]].values:
        
        tmpL = len(str(i[0]).split(','))
        if tmpL>maxN:
           maxN = tmpL

    return maxN

def add_new_columns_to_dataframe(df, colNames):
    
    df1 = df.copy()
    
    for i in colNames:
        df1[i] = pd.Series(index=df1.index)
        
    return df1
    
def split_titles_column(dataframe):
    
    df = add_new_columns_to_dataframe(data, ["title0","title1","title2","title3","title4","title5","title6","title7","title8","title9","title10"])

    for i in range(0, df.shape[0]):
#    for i in range(0, 50): 
        row = df.iloc[i,:]
        values = str(row[1]).split(',')
        for j in range(0, len(values)):
            df.iloc[i,12+j] = values[j]

    return df       

 
def split_tags_column(dataframe):
    
    df = add_new_columns_to_dataframe(data, ["tag"+str(i) for i in range(0, 57)])

    for i in range(0, df.shape[0]):
#    for i in range(0, 50): 
        row = df.iloc[i,:]
        values = str(row[10]).split(',')
        for j in range(0, len(values)):
            df.iloc[i,23+j] = values[j]
    return df     
    
data=pd.read_csv('./Data/item_profile.csv', sep='\t', header=0, usecols=['id', 'career_level', 'discipline_id', 'industry_id', 'country', 'region', 'latitude', 'longitude', 'employment', 'active_during_test', 'tags', 'title'])

df = split_titles_column(data)
df = split_tags_column(data)

print(df.iloc[:,12:])    

df.to_csv("item_profile_splited.csv", sep='\t')