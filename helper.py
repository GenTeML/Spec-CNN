# -*- coding: utf-8 -*-
'''
Contains functions for managing, shaping, and modifying data or analysis
'''

import numpy as np
import pandas as pd

def fnamelsbuilder(fin_path):
  """Build a list of files in directory 'fin_path'.

  Args:
    fin_path: a string providing the path to the folder with the intended files

      In order to avoid unexpected behavior, ensure the fin_path folder only
      contains folders or data files

  Returns:
    A python list of file names in fin_path
  """

  from os import listdir
  from os.path import isfile, join
  #return files
  return [f for f in listdir(fin_path) if isfile(join(fin_path,f))]

def peakscleaning(df):
  """Cleaning for peaks data - drop any rows containing NA, scale features

  Args:
    df: a dataframe with peaks data

  Returns:
    DataFrame of peaks data with no NA values, features scaled with sklearn
    preprocessing.StandardScaler
  """
  df.dropna(inplace=True)

  #drop relative intensities
  df.drop(columns=[i for i in df.columns.values if 'val' in i],inplace=True)
  return df

def dfbuilder(fin_path,split_df=True,dev_size=.2,r_state=1,raw=False):
  """Imports data from all CSV files in 'fname_ls' found at location 'fin_path'
  and returns in one large dataframe or a split of data for training

  Args:
    fin_path: string, path to the folder with the intended files
    split_df: boolean, true causes 'df_builder' to return split data df;
      default True. If True, function will return split data; if False, function
      will return a single DataFrame
    dev_size: float on closed interval (0, 1.0), determines percentage of data
      used for the dev set in the train_test_split. Ignored if 'split_df' is
      False
    r_state: integer, provides random state for the train_test_split. Ignored if
      'split_df' is False
    raw: boolean, True if the input is raw data, false if it has been
      preprocessed (eg. with continuous wavelet transform)

  Returns:
    If split_data=True, Tuple of 4 DataFrames including all rows from files
    named in fname_ls split using the 'split_data()' function.
    If 'split_df' is False, will return one DataFrame of data in fin_path
  """

  #list of file names with data
  fname_ls=fnamelsbuilder(fin_path)

  #create list to hold dataframes
  df_ls=[]
  #read in each file
  if raw:
    df_ls=raw_processing(df_ls,fname_ls,fin_path)

  else:
    for i in fname_ls:
      temp_df=pd.read_csv(fin_path+i,index_col=0)
      df_ls.append(temp_df)

  #create one large df
  if len(df_ls)>1:
    df=pd.concat(df_ls)
  else:
    df=df_ls[0]

  #remove rows with "None"s that will break the model
  df.dropna(axis=0,inplace=True)

  #if peaks data, additional cleaning
  if 'Peaks Only' in fin_path:
    df=peakscleaning(df)

  print('Master data set shape is',df.shape,'\n\n','Master data set is\n',df)

  #split data for processing
  if split_df:
    return splitdata(df,dev_size,r_state)

  #split data for processing
  return df

def raw_processing(df_ls,fname_ls,fin_path):
  '''imports and standardizes raw data to one intensity value per wave number
  between wave number 150 and 1100. Designed for use by the dfbuilder method

  Args:
    df_ls: an empty list for holding the imported DataFrames
    fname_ls: list of data file names
    fin_path: the path where the data files can be found

  Returns:
    A python list of DataFrames with standardized raw data files
  '''
  for fil in fname_ls:
    temp_df=pd.read_csv(fin_path+fil,index_col='og-idx',delim_whitespace=False)

    #separate the labels
    temp_labels=temp_df['label']
    temp_df.drop(columns=['label'],inplace=True)

    #trim to cols to [150,1100]
    trim_range=(150.,1100.)

    temp_df.drop(columns=[j for j in temp_df.columns.values if float(j)<trim_range[0]],inplace=True)
    temp_df.drop(columns=[j for j in temp_df.columns.values if float(j)>trim_range[1]],inplace=True)

    #standardize to 1 intensity value per wave number
    std_df=pd.DataFrame()

    for k in range(int(trim_range[0]),int(trim_range[1])):
      std_df[k]=temp_df[[j for j in temp_df.columns.values if k<=float(j)<(k+1)]].min(axis=1)

    #add labels back to DataFrame, append to the df_ls DataFrame list
    std_df['label']=temp_labels.values
    df_ls.append(std_df)

  return df_ls

def splitdata(X,dev_size=0.2,r_state=1):
  """splits X values from y values and returns tuple of DataFrames from sklearn
  train_test_split

  Args:
    X: DataFrame or ndarray with labels in the last column
    dev_size: float on closed interval (0, 1.0), determines percentage of data
      used for the dev set in the train_test_split
    r_state: integer, provides random state for the train_test_split

  Returns:
    Tuple of 4 DataFrames from 'X' split using sklearn train_test_split
  """

  #separate y from X
  y=X[X.columns[-1]]
  X.drop(X.columns[-1],axis=1,inplace=True)
  #split into train and dev sets
  from sklearn.model_selection import train_test_split
  return train_test_split(X,y,test_size = dev_size,random_state = r_state)

def plot_roc(X_df,i):
  """plots the receiver operating characteristic curve for the data in X_df with
  true binary labels in the final column

  Args:
    X_df: DataFrame with the target class probability in the column ['i']
    i: target integer label, eg use 0 to get an ROC curve for Quartz

  Returns:
    a tuple of arrays, the arrays of tpr, fpr, and threshold values

  Notes:
    The Reciever Operator Characteristic (ROC) curve is built by plotting x,y at
    various binary discrimination thresholds where x=true positive rate(tpr) and
    y=false positive rate(fpr)

    tpr=True positive/(True Positive + False Negative)
    fpr=False positive/(False Positive + True Negative)
  """
  #get tpr, fpr, and threshold lists
  try:
    from sklearn.metrics import roc_curve
    fpr_p,tpr_p,thresh=roc_curve(X_df['label'],X_df[i],i)
  except:
    return None

  #plot the roc curves
  from matplotlib import pyplot as plt
  plt.plot(fpr_p,tpr_p)
  plt.plot([0,1],[0,1],color='green')
  plt.title('ROC Curve for Class '+str(i))
  plt.show()


  return tpr_p,fpr_p,thresh

def roc_all(outputs,labels):
  """creates ROC curves for each class in the output of a classifier

  Args:
    outputs: DataFrame or ndarray of output probabilities for each class
    labels: an array or series of true labels for the samples in X

  Returns:
    None
  """
  #print('outputs:\n\n',outputs)
  #print('\n\nlabels:\n\n',labels)
  master_df=pd.DataFrame(outputs)
  roc_d={}
  master_df['label']=labels.values
  for i in range(labels.max()+1):
    #print('target values is',i)
    if len(master_df.loc[master_df['label']==i])==0:
      continue
    roc_d[i]=plot_roc(master_df,i)

  return pd.DataFrame(roc_d,index=['tpr','fpr','thresh'])

def dec_pred(y_pred,threshold=0.95):
  """takes prediction weights and applies a decision threshold to deterime the
  predicted class for each sample

  Args:
    y_pred: an ndarray of prediction weights for a set of samples
    threshold: the determination threshold at which the model makes a prediction

  Returns:
    a 1-d array of class predictions, unknown classes are returned as class 6
    """
  import numpy as np
  probs_ls=np.amax(y_pred,axis=1)
  class_ls=np.argmax(y_pred,axis=1)
  pred_lab=np.zeros(len(y_pred))
  for i in range(len(probs_ls)):
    if probs_ls[i]>threshold:
      pred_lab[i]=class_ls[i]
    else:
      pred_lab[i]=6
  return pred_lab
