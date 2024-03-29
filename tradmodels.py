'''
Contains methods that define the traditional models
'''
import pandas as pd
import numpy as np
import helper as h

def scale_data(pk_df):
  #scale data((X-mean)/std_dev)
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X = sc.fit_transform(pk_df)
  return X

def set_split(target_set,noise_set):
  #get the size of the noise_set for use in the model
  size=int((target_set.shape[0]/noise_set.shape[0])*1.2*noise_set.shape[0])

  #randomly select 1.2x the target_set rows from the noise set
  from numpy.random import default_rng
  msk = default_rng(seed=1).uniform(0,1,len(noise_set)) < size
  noise_set = noise_set[msk]
  noise_set.label=0
  target_set.label=1
  return h.splitdata(noise_set.append(target_set),dev_size=0.2,r_state=1)

def train_model(X_train,y_train,mtype):
  #create model depending on the mtype passed
  if mtype == r'Binary Model':
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
  if mtype == r'Logistic':
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500).fit(X_train,y_train)
  if mtype == r'SVC':
    from sklearn.svm import LinearSVC
    return LinearSVC(penalty='l1',dual=False,max_iter=5000).fit(X_train,y_train)
  else:
    return None

def assess_model(model,X_dev,y_dev,m_label,fil_id,thresh,save_path):
  try:
    preds=model.predict_proba(X_dev)
  except:
    preds=model.decision_function(X_dev)

  plot_confusion_matrix(y_dev,np.argmax(preds,axis=1),'Confusion Matrix',save_path,m_label)

  output_probs=pd.DataFrame(preds,index=y_dev.index.values)
  output_probs['label']=y_dev
  output_probs.to_csv(save_path+'/'+str(m_label)+'_probs.csv',mode='w')

def run_pca(df):
  temp_df=df.drop(['label'],axis=1)

  from sklearn.decomposition import PCA
  pca=PCA(0.8)
  output_df=pd.DataFrame(pca.fit_transform(temp_df.values),index=df.index.values)
  output_df['label']=df['label']
  return output_df,pca

def train_model_set(fin_path,fil_id,save_path):
  """Trains a binary classifier model for each class in the training set and
  returns a list of those binary_classifier_probs

  Args:
    fin_path: string, path to the folder with the intended filepaths

  Returns:
    a tuple with the following: list of scikit-learn random forest models, list
    of labels included in the training set, and a list of PCA models for each
    label
  """

  #build master dataframe of data from fin_path and run pca to reduce the feature set
  df,pca=run_pca(h.dfbuilder(fin_path,split_df=False,dev_size=.2,r_state=1,raw=False))

  #list for holding models
  m_list=[]

  #loop to create one model for each label in df
  l_list=sorted(df['label'].unique().tolist())
  for i in l_list:
    #print('in loop, label is',i,'\n\n')
    #for label i, select the target label set and create a training and dev set
    target_df=df.loc[df['label']==i,:]
    noise_df=df.loc[df['label']!=i,:]
    X_train,X_dev,y_train,y_dev=set_split(target_df,noise_df)
    m_list.append(train_model(X_train,y_train.values.ravel(),'Binary Model'))
    #print('model trained')
    assess_model(m_list[-1],X_train,y_train,i,fil_id=fil_id+'_train',thresh=0.0,save_path=save_path)
    assess_model(m_list[-1],X_dev,y_dev,i,fil_id=fil_id+'_dev',thresh=0.0,save_path=save_path)
  return m_list,l_list,pca

def make_model_dir(fil_id,mtype):
  from os import mkdir,path,getcwd
  working_dir=getcwd()
  save_path=(path.join(working_dir,r'Model Data',mtype,fil_id,))
  mkdir(save_path)
  return save_path

def plot_confusion_matrix(y_true, y_pred, title, save_path, m_label):
  import matplotlib.pyplot as plt
  import seaborn as sn
  from sklearn.metrics import confusion_matrix
  labels = np.arange(15) #15 is the number of labels currently
  cm = confusion_matrix(y_true, y_pred, labels)
  cm_df = pd.DataFrame(cm, columns = labels, index = labels)
  cm_df.to_csv(save_path+str(m_label)+'_confmat.csv',mode='w')
  cm_df.columns.name = 'Predicted'
  cm_df.index.name = 'Actual'
  sn.heatmap(cm_df, annot = True, cmap = 'Blues', fmt = 'd')
  plt.title(title)
  plt.show()

def prep_test(pca,testin_path=r'/Data/CWT Data/Single/',r_state=1,raw=False):
  init_test_df=h.dfbuilder(testin_path,split_df=False,r_state=1,raw=raw)
  init_test_df.dropna(inplace=True)
  test_df=pd.DataFrame(pca.transform(init_test_df.drop(['label'],axis=1).values),index=init_test_df.index.values)
  test_df['label']=init_test_df['label']
  return test_df

def binary_model_set(fin_path=r'Data/CWT Data/Single/',testin_path=r'/Data/CWT Data/Mixed/',fil_id='0',raw=False,r_state=1):
  """Trains a series of binary models, one for each label in the data set found
  in fin_path and tests against data in the testin_path

  Args:
    fin_path: string, path to folder with training data files
    testin_path: string, path to folder with the test data
    raw: boolean, True if the input is raw data, false if it has been Preprocessed
    r_state: int, random seed value

  Returns:
    Tuple of objects in the following order: DataFrame of predicted labels, a
    list of trained binary models, a list of trained PCA models
  """

  #get the current working path to provide a place to save the model data
  save_path=make_model_dir(fil_id,r'Binary Model')

  #train the models for testing
  m_list,l_list,pca=train_model_set(fin_path,fil_id,save_path)

  #import test data, perform pca, re-associate labels
  test_df=prep_test(pca,testin_path,r_state,raw)

  #create an empty ndarray to hold the predicted labels
  labels=np.empty(shape=(len(test_df.index),len(m_list)))

  #test each of the binary classifiers
  for j in range(len(m_list)):
    labels[:,j]=m_list[j].predict_proba(test_df.drop(['label'],axis=1))[:,1]
  label_df=pd.DataFrame(labels,index=test_df.index.values,columns=l_list)
  label_df['label']=test_df['label']

  #save model data
  label_df.to_csv(save_path+'/'+'binary_classifier_probs_'+fil_id+'.csv',mode='w')

  return label_df,m_list,pca

def define_model(fin_path,fil_id,save_path,mtype):
  """Runs PCA and trains a model of the selected type (either SVC or Logistic
  regression)

  Returns:
    Tuple of objects in the following order: a trained scikit-learn logistic
    regression model, a trained scikit-learn pca model
  """
  #build master dataframe of data from fin_path and run pca to reduce the feature set
  df,pca=run_pca(h.dfbuilder(fin_path,split_df=False,dev_size=.2,r_state=1,raw=False))

  X_train,X_dev,y_train,y_dev=h.splitdata(df,0.2,1)

  #train model
  tmodel=train_model(X_train,y_train,mtype)
  print('\n\nValidation\n')
  assess_model(tmodel,X_dev,y_dev,mtype,fil_id=fil_id+'_dev',thresh=0.0,save_path=save_path)
  from sklearn.metrics import classification_report

  cr = classification_report(y_test, np.argmax(model.predict(X_test), axis = -1), output_dict = True)
  cr_df = pd.DataFrame(cr).transpose()
  return sorted(df['label'].unique().tolist()),tmodel,pca

def classic_model(fin_path=r'Data/CWT Data/Single/',testin_path=r'/Data/CWT Data/Mixed/',fil_id='0',raw=False,r_state=1,mtype=r'Logistic'):
  """Trains a logistic regression model, one for each label in the data set
  found in fin_path and tests against data in the testin_path

  Args:
    fin_path: string, path to folder with training data files
    testin_path: string, path to folder with the test data
    raw: boolean, True if the input is raw data, false if it has been Preprocessed
    r_state: int, random seed value
    mtype: string, type of model - 'Logistic' for logistic regression or 'SVC'
    for the SVC model

  Returns:
    Tuple of objects in the following order: A logistic regression model, a
    trained PCA model
  """

  #get the current working path to provide a place to save the model data
  save_path=make_model_dir(fil_id,mtype)

  #train the models for testing
  y,cmodel,pca=define_model(fin_path,fil_id,save_path,mtype)

  #import test data, perform pca, re-associate labels
  test_df=prep_test(pca,testin_path,r_state,raw)

  #test the model
  try:
    labels=cmodel.predict_proba(test_df.drop(['label'],axis=1))
  except:
    labels=cmodel.decision_function(test_df.drop(['label'],axis=1))
  label_df=pd.DataFrame(labels,index=test_df.index.values,columns=y)
  label_df['label']=test_df['label']

  #save model data
  label_df.to_csv('Model Data/'+mtype+'/probs_'+fil_id+'.csv',mode='w')

  return label_df,cmodel,pca
