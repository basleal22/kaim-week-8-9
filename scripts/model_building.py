import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import SimpleRNN,LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def logistic_regression(x_train,y_train,x_test,y_test):
    model_1 = LogisticRegression(class_weight='balanced')
    model_1.fit(x_train,y_train)
    predict = model_1.predict(x_test)
    return predict
def randomforest(x_train,y_train,x_test,y_test):
    model_2 = RandomForestClassifier(class_weight='balanced',n_estimators=100)
    model_2.fit(x_train,y_train)
    predict = model_2.predict(x_test)
    return predict
def gradientboost(x_train,y_train,x_test,y_test):
    model_3 = HistGradientBoostingClassifier(class_weight='balanced')
    model_3.fit(x_train,y_train)
    predict=model_3.predict(x_test)
    return predict
def cnnmodel(x_train,y_train,x_test,y_test):
    x_train = x_train.to_numpy().reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.to_numpy().reshape(x_test.shape[0], x_test.shape[1], 1)
    model_4 = Sequential([layers.Conv1D(32,2,activation = 'relu',input_shape=(x_train.shape[1],1)),
                          layers.Conv1D(64,2,activation='relu'),
                          layers.Flatten(),
                          Dense(128,activation='relu'),
                          Dense(1,activation='sigmoid')])
    model_4.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_4.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_test,y_test), callbacks=[early_stop])
    prediction=model_4.predict(x_test)
     # Convert probabilities to binary (threshold = 0.5)
    y_pred_binary = (prediction >= 0.5).astype(int)
    return y_pred_binary



