import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import (f1_score, roc_auc_score, precision_recall_curve, 
                            roc_curve, confusion_matrix, classification_report, 
                            accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier



def heatmap_count(df, var1, var2):
  """Generates a Count heatmap of any two columns of a data frame

  Parameters
  ----------
  df : DataFrame
      A DataFrame that has column features
  var1 : str
      The name of a column in the DataFrame
  var2 : str
      The name of a column in the DataFrame. Different from var1
  
  Returns
  -------
  plot : seaborn.heatmap
      A heatmap of the 2 variables grouped by count, with preset configuration
  """

  heat_df=[]
  heat_df=pd.DataFrame(df.groupby([var1,var2]).count().iloc[:,1]).reset_index()
  cols=heat_df.columns
  value=[y for y in cols if y not in [var1, var2]]
  heat_df=heat_df.pivot(columns=var1, index=var2, values=value[0])
  plt.figure(figsize=(15,5))
  return sns.heatmap(heat_df, cmap="Blues",linewidths=0.5, annot=True, fmt='g')


def get_X_y(df, y_name):
  """
  Splits a DataFrame in X (Features) and y (response variable)
  Parameters
  ----------
  df : DataFrame
      A DataFrame that has column features.
  y_name : str
      The name of the target varriable in the DataFrame.

  Returns
  -------
  X : DataFrame
      DataFrame with the features used to predict y.
  y : Array(int)
      Array with the response variable's values.
  """
  y=[y_name]
  X=[col for col in df.columns if col not in y]
  y=df[y].copy().values.flatten()
  X=pd.get_dummies(df[X].copy())
  return X, y



def data_preprocessing_up_or_down_sample(X, y, sample="up", test_size=0.2):
  """
  Applies downsampling or upsampling, and returns the Train-Test split 
  of data.
  Parameters
  ----------
  X : DataFrame
      DataFrame with the features used to predict y.
  y : Array(int)
      Array with the response variable's values.
  sample : str(optional)
      Chooses the method to apply. Downsampling ("down"), 
      upsampling ("up") or no method (anything else). Default is "up". 
  test_size : float(optional)
      Sets the test_size parameter of sklearn.model_selection.train_test_split.
      Default is 0.2
  
  Returns
  -------
  splitting : list, length=2 * len(arrays)
      List containing train-test split of inputs, with the method defined by
      the parameter sample applied.
  """

  # Use the sample parameter to define local variables to select the correct 
  # method
  a,b=0,0
  if sample=="up": 
    a,b=1,0
  if sample=="down":
    a,b=0,1 

  
  # Apply the normal train_test_split to the data
  X_train, X_test, y_train, y_test = train_test_split( X, y, \
                                        test_size=test_size)
  # Using the a and b local variables, apply downsampling or upsampling only
  # if the sample parameter is "up" or "down".

  if a+b>=1:
    X_train_temp, y_train_temp = resample(X_train[y_train == a],
                                    y_train[y_train == a],
                                    n_samples=X_train[y_train == b].shape[0])
    X_train = np.concatenate((X_train[y_train == b], X_train_temp))
    y_train = np.concatenate((y_train[y_train == b], y_train_temp))
  return (X_train, X_test, y_train, y_test)



def rocauc_plot(model, model_name, y_test, X_test):
  """
  Plots the ROC-AUC curve for a model. 
  Multiple consecutive calls will allow to display multiple curves on the
  same plot

  Parameters
  ----------
  model : estimator object.
      Either from sklearn or keras interfaces
  model_name : str
      Name of the plot
  X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
      Input values
  y_test : {array-like, sparse matrix} of shape (n_samples, n_features)
      Target values
  test_size : float(optional)
      Sets the test_size parameter of sklearn.model_selection.train_test_split.
      Default is 0.2
  
  Returns
  -------
  plot : matplotlib.pyplot
      Plot of the ROC-AUC score
  """
  try:
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
  except:
    auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
  plt.plot(fpr, tpr, label=model_name+" AUC = {:.5f}".format(auc))
  plt.title("Curva(s) ROC", fontdict={"fontsize": 21})
  plt.xlabel("False positive rate", fontdict={"fontsize": 13})
  plt.ylabel("True positive rate", fontdict={"fontsize": 13})
  plt.legend(loc="lower right")
  plt.plot([0, 1], [0, 1], "r--")



def plot_roc_conf_matrix(y_test,X_test, model, model_name):
  """
  Print the classification report and plots the ROC-AUC curve and confusion 
  matrix for a given model.

  Arguments
  ---------
  y_test : {array-like, sparse matrix} of shape (n_samples, n_features)
      Target values
  X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
      Input values
  model : estimator object.
      Either from sklearn or keras interfaces
  model_name : str
      Name of the model. Used as part of the plot's titles.
  
  """
  try:
    y_pred=model.predict_classes(X_test)
  except:
    y_pred=model.predict(X_test)
  cm = metrics.confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
  plt.title(model_name+ " - Matriz de confusión", y=1.1, \
            fontdict={"fontsize": 21})
  plt.xlabel("Predicted", fontdict={"fontsize": 14})
  plt.ylabel("Actual", fontdict={"fontsize": 14})
 
  print(classification_report(y_test, y_pred))
  plt.subplot(1,2,2)

  rocauc_plot(model, model_name, y_test, X_test)



def apply_model_to_df(data, model, model_name):
  """
  Applies a sklearn estimator to a DataFrame. 
  It returns a fitted model and shows relevant information of the model's
  performance (applies the plot_roc_conf_matrix function to it)

  Parameters
  ----------
  data : List of Arrays
      List of arrays, equivalent to the output of the function
      sklearn.model_selection.train_test_split  
  model : estimator object.
      Either from sklearn or keras interfaces
  model_name : str
      Name of the model. Used as part of the plot's titles.
  
  Returns
  -------
  model : estimator instance
      Fitted classifier or a fitted Pipeline in which the last estimator 
      is a classifier.
  """
  X_train, X_test, y_train, y_test=data
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  plot_roc_conf_matrix(y_test,X_test, model, model_name)
  return model
  
  
  
def pipeline_classifier(X,y,model, param_grid):
  """
  Creates a general Pipeline for sklearn classifiers.
  Applies GridSearchCV to optimize hyper parameters of the model.

  Parameters
  ----------
  X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Input values
  y : {array-like, sparse matrix} of shape (n_samples)
      Target values
  model : sklearn.estimator object
      Instance of an estimator form sklearn.
  param_grid: dict or list of dictionaries
      Dictionary with parameters names (str) as keys and lists of parameter 
      settings to try as values, or a list of such dictionaries, in which case 
      the grids spanned by each dictionary in the list are explored. 
      This enables searching over any sequence of parameter settings.
    
  Returns
  -------
  model : sklearn.estimator instance
      Fitted classifier or a fitted Pipeline in which the last estimator 
      is a classifier.
  """
  pipe = make_pipeline(StandardScaler(), model)
  clf= GridSearchCV(pipe,
                    param_grid=param_grid,
                    cv=10,
                    refit=True,
                    scoring="f1",
                    n_jobs=-1)
  clf.fit(X,y)
  return clf
  

def nn_base_model():
  """
  A basic wrapper function to use Keras NN on GridSearchCV.
   
  Returns
  -------
  nn : wrapped keras.Sequential model 
      A Keras basic Neural Network for Binary classification, wrapped for usage
      in KerasClassifier() build_fn parameter.
      
  """
  nn=Sequential()
  nn.add(Dense(20, input_dim=20, activation='relu'))
  nn.add(Dropout(0.15))
  nn.add(Dense(1, activation='sigmoid'))
  nn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
  return nn
 
