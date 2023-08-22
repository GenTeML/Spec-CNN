# -*- coding: utf-8 -*-

'''
Contains methods that define the model with preprocessed CWT data
'''
import utils.helper as h
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, LeakyReLU
from tensorflow.keras.layers import MaxPooling1D, Dropout
from tensorflow.keras.models import Model

class PredictionCallback(tf.keras.callbacks.Callback):
  '''
  Callback to output list of predictions of all training and dev data to a file after each epoch
  '''
  def __init__(self,train_data,validation_data,y_train,y_dev,train,val):
    self.validation_data=validation_data
    self.train_data=train_data
    self.y_train=y_train
    self.y_dev=y_dev
    self.train=train
    self.val=val
  def on_epoch_end(self, epoch, logs=None):
    train=r'Model Data/CNN Model/CWT/'+self.train
    try:
      pd.DataFrame(data=self.model.predict(self.train_data),index=self.y_train.index).to_csv(train,mode='a')
    except:
      pd.DataFrame(data=self.model.predict(self.train_data),index=self.y_train.index).to_csv(train,mode='w')

    val=r'Model Data/CNN Model/CWT/'+self.val
    try:
      pd.DataFrame(data=self.model.predict(self.validation_data),index=self.y_dev.index).to_csv(val,mode='a')
    except:
      pd.DataFrame(data=self.model.predict(self.validation_data),index=self.y_dev.index).to_csv(val,mode='w')

def train_cnn_model(X_train,y_train,X_dev,y_dev,hyperparameters=None,fast=True,id_val=None):
  '''
  Trains a CNN model using the predetermined architecture and returns the model

  Args:
    X_train: DataFrame or ndarray of training set input features
    y_train: DataFrame or Series of labels for the X_train samples, must be in
      the same order as X_train samples
    X_dev: DataFrame or ndarray of dev set input features
    y_dev: DataFrame or Series of labels for the X_dev samples, must be in the
      same order as the X_train samples
    hyperparameters: an array of hyperparameters in the order: lr (learning
      rate)(default 0.001), batch size (default 100), drop (dropout rate)
      (default 0.55), epochs (default 100) - this will be updated in a future
      state after the current deadline (5/25/2020)
    fast: boolean. if true, then the model runs more than twice as fast but does
      not record predictions of every sample for every epoch. If False, runs
      much more slowly but creates a csv of probability weights for every sample
      at every epoch for both the training and validation sets
    id_val: string that is prepended to all output files for tracking

  Returns:
      a tuple containing the trained model and the History object from training
      that model
  '''

  #if no hyperparameters are set, set the defaults, otherwise extract them
  if not hyperparameters:
    lr=0.001
    batch_size=500
    drop=0.55
    epochs=5
  else:
    lr=hyperparameters[0]
    batch_size=hyperparameters[1]
    drop=hyperparameters[2]
    epochs=hyperparameters[3]

  #determine the appropriate callbacks, depening on if fast is true or false
  if not fast:
    callbacks=[PredictionCallback(X_train,X_dev,y_train,y_dev,id_val+'train_outputs.csv',
                                    id_val+'val_outputs.csv'),
              K.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=3)]
  else:
    callbacks=[K.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=3)]

  #call build_cnn and train model, output trained model
  cnn_model=build_cnn(X_train.shape,y_train.max()+1,lr,drop)

  mout_path=r'Data/Model Data/CWT CNN/'

  cnn_hist=cnn_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_dev,y_dev),callbacks=callbacks)

  return cnn_model,cnn_hist

def layer_CBnAP(X_in,nfilters,size_C,s_C,size_P,lnum):
  '''Defines one set of convolutional layers with a Convolution,
  BatchNormalization, LeakyReLU activation, and MaxPooling1D

  Args:
    X_in: input matrix for the layers
    nfilters: number of filters for the convolutional layer
    size_C: size of the convolutional kernel
    s_C: step size for the convolution layer
    size_P: size for MaxPooling layer
    lnum: layer number used for debugging

  Returns:
    Matrix of output values from this set of layers
  '''

  # CONV -> BN -> RELU -> MaxPooling Block applied to X
  X_working=Conv1D(nfilters,size_C,s_C,name='conv'+lnum)(X_in)
  X_working=BatchNormalization(name='bn'+lnum)(X_working)
  X_working=LeakyReLU(alpha=0.3,name='relu'+lnum)(X_working)
  X_working=MaxPooling1D(size_P,name='mpool'+lnum)(X_working)
  return X_working

def build_cnn(X_shape,y_shape,lr=0.001,drop=0.55):
  '''Defines and builds the CNN model with the given inputs

  Args:
    X_shape: the shape of the data for the model
    y_shape: the shape of the labels for the model
    lr: learning rate, default 0.001
    drop: drop rate, default 0.55

  Returns:
    a compiled model as defined by this method
  '''
  #set the output path for model data
  mout_path=r'Data/Model Data/CWT CNN/'

  # Define the input placeholder as a tensor with the shape of the features
  #this data has one-dimensional data with no channels
  X_input=Input((X_shape[1],1))

  #first layer - conv, batch normalization, activation, pooling
  nfilters=16
  size_C=21
  s_C=1
  size_P=2
  X=layer_CBnAP(X_input,nfilters,size_C,s_C,size_P,'1')

  #second layer - conv, batch normalization, activation, pooling
  nfilters=32
  size_C=11
  s_C=1
  size_P=2
  X=layer_CBnAP(X,nfilters,size_C,s_C,size_P,'2')

  #third layer - conv, batch normalization, activation, pooling
  nfilters=64
  size_C=5
  s_C=1
  size_P=2
  X=layer_CBnAP(X,nfilters,size_C,s_C,size_P,'3')

  #flatten for final layers
  X=Flatten()(X)

  #layer 4 - fully connected layer 1 dense,Batch normalization,activation,dropout
  X=Dense(2048, use_bias=False,name='dense4')(X)
  X=BatchNormalization(name='bn4')(X)
  X=Activation("tanh",name='tanh4')(X)
  X=Dropout(drop,name='dropout4')(X)

  #layer 5 - fully connected layer 2 dense, batch normalization, softmax output
  X=Dense(y_shape,use_bias=False,name='dense5')(X)
  X=BatchNormalization(name='bn5')(X)
  outputs=Activation("softmax",name='softmax5')(X)

  model=Model(inputs=X_input,outputs=outputs)

  opt=K.optimizers.RMSprop(learning_rate=lr)
  #opt=K.optimizers.Nadam(lr)
  model.summary()
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=opt,metrics=['sparse_categorical_accuracy'])
  return model

def test_cnn_model(model,X_test,y_test,id_val='0',test=True,threshold=0.95,fast=False):
  """
  test a trained model with given parameters, creates a csv of confusion matrix
  at Model Data/CNN Model/ 'id_val'+comfmatout.csv

  Args:
    model: a trained keras model used to predict the categories of the test set
    X_test: a DataFrame or ndarray of sample features sets for testing
    y_test: a DataFrame or Series of sample labels the X_test feature set - must
      be in the same order as the X_test feature set
    id_val: a string used in the file name of the outputs for identification

  Returns:
    None; creates a file at the /Model Data/CNN Model/CWT/ folder
  """
  #predict classes for provided test set
  y_pred=model.predict(X_test,batch_size=1)

  #report confusion matrix
  confmat=build_confmat(y_test,y_pred,threshold)
  display(confmat)

  #if fast is not True, save the confusion matrix as either test or validation
  if not fast:
    if test:
      id_val=id_val+'_test'
    else:
      id_val=id_val+'_validation'
    #save confusion matrix as csv to drive
    confmatout_path=r'Model Data/CNN Model/CWT/'+id_val

    confmat.to_csv(confmatout_path+r'_confmat.csv')
    #save output weights
    pd.DataFrame(data=model.predict(X_test),index=y_test.index.values).to_csv(confmatout_path+'_probs.csv')

def save_model(model,mout_path):
  '''saves model data to the given output mout_path

  Args:
    model: the model history file
    mout_path: the filepath to store the model dataframe

  Returns:
    None. Creates a file at the given filepath
  '''
  model.save(mout_path+'cnn.h5')
  print('model saved')

def dec_pred(y_pred,threshold=0.95):
  """takes prediction weights and applies a decision threshold to deterime the
  predicted class for each sample

  Args:
    y_pred: an ndarray of prediction weights for a set of samples
    threshold: the determination threshold at which the model makes a prediction

  Returns:
    a 1-d array of class predictions, unknown classes are returned as class 8
    """
  import numpy as np
  probs_ls=np.amax(y_pred,axis=1)
  class_ls=np.argmax(y_pred,axis=1)
  pred_lab=np.zeros(len(y_pred))
  for i in range(len(probs_ls)):
    if probs_ls[i]>threshold:
      pred_lab[i]=class_ls[i]
    else:
      pred_lab[i]=15
  return pred_lab

def build_confmat(y_label,y_pred,threshold):
  '''builds the confusion matrix with labeled axes

  Args:
    y_label: a list of true labels for each sample
    y_pred: a list of predicted labels for each samples
    threshhold: the decision threshhold for the mat_labels

  Returns:
    A DataFrame containing the confusion matrix, the column names are the
    predicted labels while the row indices are the true labels
  '''
  _y_pred=dec_pred(y_pred,threshold)

  mat_labels=range(max([max(y_label),int(max(_y_pred))])+1)

  from sklearn.metrics import confusion_matrix
  return pd.DataFrame(confusion_matrix(y_label,_y_pred,mat_labels),index=['true_{0}'.format(i) for i in mat_labels],columns=['pred_{0}'.format(i) for i in mat_labels])

def cwt_cnn_model(fin_path=r'Data/CWT Data/Single/',mout_path=r'Model Data/CNN model/CWT/',dev_size=0.2,r_state=1,hyperparameters=None,fast=True,fil_id='0',threshold=.98):
  '''calls methods to build and train a model as well as testing against the
  validation sets

  Args:
    fin_path: file path for pulling in data
    mout_path: file path for saving model data
    dev_size: size of the dev set as a percentage between 0 and 1 inclusive.
      Default 0.2
    r_state: random seed value. Default 1
    hyperparameters: an array of hyperparameters in the order: learning
      rate (lr) (default 0.0001), batch size (default 100), dropout rate (drop))
      (default 0.55), epochs (default 10)
    fast: boolean. if true, then the model runs more than twice as fast but does
      not record predictions of every sample for every epoch. If False, runs
      much more slowly but creates a csv of probability weights for every sample
      at every epoch for both the training and validation sets
    id_val: string that is prepended to all output files for tracking
    threshold: decision threshold for labeling

  Returns:
    CNN model built with preprocessed CWT data inputs
  '''

  #build dataframes for all data after splitting
  X_train,X_dev,y_train,y_dev=h.dfbuilder(fin_path=fin_path,dev_size=dev_size,r_state=r_state,
                                          raw=False)

  #train a cnn model - v1.0
  cnn_model,cnn_hist=train_cnn_model(X_train,y_train,X_dev,y_dev,hyperparameters,fast,fil_id)
  pd.DataFrame(cnn_hist.history).to_csv(mout_path+fil_id+'hist.csv')

  #test cnn model with dev set
  test_cnn_model(cnn_model,X_dev,y_dev,test=False,threshold=threshold)

  #save model
  if not fast:
    save_model(cnn_model,mout_path+fil_id)

  return cnn_model,cnn_hist,X_dev,y_dev
