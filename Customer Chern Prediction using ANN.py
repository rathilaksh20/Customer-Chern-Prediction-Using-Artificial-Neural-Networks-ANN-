import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\rathi\Downloads\Churn_Modelling.csv')
print(df.head())
print(df.info())

#Checking whether is there any duplicated row or not
df.duplicated().sum()

#Checking how many customers exited bank
print(df['Exited'].value_counts())

#Checking geography of customers
print(df['Geography'].value_counts())

#Removing unnessary columns
df.drop(columns=['CustomerId','Surname', 'RowNumber'], inplace = True)

print(df.head())

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)


print(df)

#It is better to scale all the values in every column so that it converges to some point like 2 digit number or 5 digit number or etc
X = df.drop(columns = ['Exited'])
y = df['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)

import tensorflow
import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
#The summary tells us that in total we have x no of trainable parameters

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

print(model.layers[0].get_weights())
y_log = model.predict(X_test_scaled)
y_pred = np.where(y_log>0.5, 1, 0)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy", "Validation Accuracy"])
plt.show()
