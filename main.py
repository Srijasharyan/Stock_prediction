import yfinance as yf
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from collections import deque
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import datetime as dt

# function to get stock price within given interval
def print_data(ticker,start,end):
  df = yf.Ticker(ticker).history(start=start,end=end)
  if "date" not in df.columns:
    df["date"] = df.index
  df=df.drop(['Dividends','Stock Splits'],axis=1)
  return df



#function to process raw data into simplified data
def process_data(ticker, feature_columns,start,end, n_steps, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    def shuffle_in_unison(a, b):
      # shuffle two arrays in the same way
      state = np.random.get_state()
      np.random.shuffle(a)
      np.random.set_state(state)
      np.random.shuffle(b)
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = yf.Ticker(ticker).history(start=start,end=end)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result


# build LSTM model for predicting 
def build_model(sequence_length, n_features, units=80, cell=tf.keras.layers.LSTM, n_layers=3, dropout=0.2,loss="mean_absolute_error", optimizer="adam"):
    model =  tf.keras.Sequential()
    layers=tf.keras.layers
    i=0
    while i<n_layers:
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(layers.Dropout(dropout))
        i+=1
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model



# predict price of stocks for given no. of days
def predict_future(model, data, days,epochs,batch_size,lookup_step,N_STEPS):
    predictions=[]
    SCALE=1
    model.fit(data['X_train'],data['y_train'],epochs=epochs,batch_size=batch_size)
    last_sequence = data["last_sequence"][-N_STEPS:]
      # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # retrieve the last sequence from data
    for i in range(days):

      # get the prediction (scaled from 0 to 1)
      prediction = model.predict(last_sequence)
      # get the price (by inverting the scaling)
      if SCALE:
          predicted_price = data["column_scaler"]["Close"].inverse_transform(prediction)[0][0]
      else:
          predicted_price = prediction[0][0]
      predictions.append(predicted_price)
      x=np.delete(last_sequence[0],list(range(0, lookup_step)))
      x=np.append(x,prediction)
      x=np.reshape(x,(len(x),1))
      last_sequence=np.array([x])
    return predictions







