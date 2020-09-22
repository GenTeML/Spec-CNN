# -*- coding: utf-8 -*-
'''
Contains functions for managing, shaping, and modifying data or analysis
'''

import numpy as np
import pandas as pd

def fnamelsbuilder(fin_path,synth=False,directory='/content/drive/My Drive/ML Spectroscopy/',use_trash=False,test=False):
  """Build a list of files in directory 'fin_path'.

  Args:
    fin_path: a string providing the path to the folder with the intended files
    synth: a boolean, if True then all files will be pulled including synthetic
      data files. Use True if importing RRUFF data. 
    directory: string, the base directory of the project
      
      In order to avoid unexpected behavior, ensure the fin_path folder only 
      contains training data files and no file names include _r or _ds unless 
      they are synthetic files

  Returns:
    A python list of file names in fin_path
  """

  from os import listdir
  from os.path import isfile, join
  if test:
    #return file names that contain "_test_"
    return [f for f in listdir(directory+fin_path) if isfile(directory+join(fin_path,f)) and ('_test_' in f)]
  if synth:
    if use_trash:
      #returns all file names
      return [f for f in listdir(fin_path) if isfile(join(fin_path,f))]
    else:
      #return all file names that are not trash
      return [f for f in listdir(directory+fin_path) if isfile(directory+join(fin_path,f)) and (not ('trash_' in f or '_test_' in f))]
  #returns only file names that don't include _ds or _r  
  if use_trash:
    #return all non synth files
    return [f for f in listdir(directory+fin_path) if isfile(join(directory+fin_path,f)) and (not ('_ds' in f or '_r' in f or '_test_' in f))]
  else:
    #return all non synth non trash files
    return [f for f in listdir(directory+fin_path) if isfile(join(directory+fin_path,f)) and (not ('_ds' in f or '_r' in f or '_test_' in f or 'trash_' in f))]

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

def dfbuilder(fin_path,synth=False,split_df=True,dev_size=.2,r_state=1,directory='/content/drive/My Drive/ML Spectroscopy/',use_trash=False,raw=False,test=False):
  """Imports data from all CSV files in 'fname_ls' found at location 'fin_path' 
  and returns in one large dataframe or a split of data for training

  Args:
    fin_path: string, path to the folder with the intended files
    synth: boolean, true if synthetic data is used; default false. Select True 
      for RRUFF data
    split_df: boolean, true causes 'df_builder' to return split data df; 
      default True. If True, function will return split data; if False, function 
      will return a single DataFrame
    dev_size: float on closed interval (0, 1.0), determines percentage of data
      used for the dev set in the train_test_split. Ignored if 'split_df' is 
      False
    r_state: integer, provides random state for the train_test_split. Ignored if
      'split_df' is False

  Returns:
    Tuple of 4 DataFrames including all rows from files named in fname_ls split 
    using the 'split_data()' function. 
    If 'split_df' is False, will return one DataFrame of data in fin_path
  """

  #list of file names with data
  fname_ls=fnamelsbuilder(fin_path,synth,directory,use_trash,test=test)

  #create list to hold dataframes
  df_ls=[]
  #read in each file
  if raw:
    df_ls=raw_processing(df_ls,fname_ls,directory,fin_path)

  else:
    for i in fname_ls:
      temp_df=pd.read_csv(fin_path+i,index_col=0)
      df_ls.append(temp_df)

  #create one large df
  if len(df_ls)>1:
    df=pd.concat(df_ls)
  else:
    df=df_ls[0]
  print(df)

  #if peaks data, additional cleaning
  if 'Peaks Only' in fin_path:
    df=peakscleaning(df)

  print('Master data set shape is',df.shape,'\n\n')

  #split data for processing
  if split_df:
    return splitdata(df,dev_size,r_state)
  
  #split data for processing
  return df

def raw_processing(df_ls,fname_ls,directory,fin_path):
  t_labels={
    'qtz':0,
    'albite':1,
    'hb':2,
    'bt':3,
    'ms':4,
    'fo':5,
    'fa':6,
    'aug':7,
    'en':8,
    'an':9,
    'mc':10,
    'cal':11,
    'gyp':12,
    'hal':13
  }

  for i in fname_ls:
    if i.split('.')[-1]=='txt':
      temp_df=pd.read_csv(directory+fin_path+i,delim_whitespace=True)
      if temp_df.shape[1]<20:
        temp_df=pd.read_csv(directory+fin_path+i,sep='\t')
    else:
      temp_df=pd.read_csv(directory+fin_path+i,delim_whitespace=False)
    
    temp_df.reset_index(inplace=True)

    #create traceable index
    temp_df['fname']=i.split('.')[0]
    temp_df['og-idx']=temp_df.fname+"-"+temp_df.index.map(str)
    temp_df.set_index(keys='og-idx',drop=True,inplace=True)

    #drop non-numeric columns from data
    dropcolumns=[]
    for j in temp_df.columns.values:
      try:
        float(j)
      except ValueError:
        dropcolumns.append(j)
    temp_df.drop(columns=dropcolumns,inplace=True)

    #trim to cols to [150,1100]
    trim_range=(150.,1100.)
    
    temp_df.drop(columns=[j for j in temp_df.columns.values if float(j)<trim_range[0]],inplace=True)
    temp_df.drop(columns=[j for j in temp_df.columns.values if float(j)>trim_range[1]],inplace=True)

    #standardize wavelength
    std_df=pd.DataFrame()

    for k in range(int(trim_range[0]),int(trim_range[1])):
      std_df[k]=temp_df[[j for j in temp_df.columns.values if k<=float(j)<(k+1)]].min(axis=1)
    
    try:
      std_df['label']=t_labels[i.split('_')[0]]
    
    except:
      std_df['label']=None

    df_ls.append(std_df)
    #print('raw df:',df_ls)
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

