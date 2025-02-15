import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import SimpleRNN,LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

def logistic_regression(x_train,y_train,x_test,y_test):
    model_1 = LogisticRegression(class_weight='balanced')
    model_1.fit(x_train,y_train)
    predict = model_1.predict(x_test)
    return predict
def randomforest(x_train,y_train,x_test,y_test):
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    model_2 = RandomForestClassifier(n_estimators=100)
    model_2.fit(x_train_balanced,y_train_balanced)
    predict = model_2.predict(x_test)
    return predict
def gradientboost(x_train,y_train,x_test,y_test):
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    model_3 = GradientBoostingClassifier()
    model_3.fit(x_train_balanced,y_train_balanced)
    predict=model_3.predict(x_test)
    return predict
def cnnmodel(x_train, y_train, x_test, y_test):
    from sklearn.utils import shuffle
    # Apply SMOTE for balancing
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    # Shuffle the balanced data to ensure randomness
    x_train_balanced, y_train_balanced = shuffle(x_train_balanced, y_train_balanced)
    # Check the shape of the balanced data
    print(f"x_train_balanced shape: {x_train_balanced.shape}")
    print(f"y_train_balanced shape: {y_train_balanced.shape}")
    # Reshape for CNN input
    x_train_balanced = x_train_balanced.to_numpy().reshape(x_train_balanced.shape[0], x_train_balanced.shape[1], 1)
    x_test = x_test.to_numpy().reshape(x_test.shape[0], x_test.shape[1], 1)
    # Build the CNN model
    model_4 = Sequential([
        layers.Conv1D(32, 2, activation='relu', input_shape=(x_train_balanced.shape[1], 1)),
        layers.Conv1D(64, 2, activation='relu'),
        layers.Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_4.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_4.fit(x_train_balanced, y_train_balanced,epochs=20,batch_size=32,validation_data=(x_test,y_test), callbacks=[early_stop])
    prediction=model_4.predict(x_test)
     # Convert probabilities to binary (threshold = 0.5)
    y_pred_binary = (prediction >= 0.5).astype(int)
    return y_pred_binary



