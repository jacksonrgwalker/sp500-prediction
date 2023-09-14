# Load in packages
import numpy as np
import pandas as pd
from pathlib import Path
from keras import Sequential
from keras.layers import Dense, Dropout, Activation

# Create a neural net class
class NeuralNetwork():
    def __init__(self, layers, neurons, dp):
        self.model = Sequential()
        self.layers = layers
        self.nodes = neurons
        self.drop_percent = dp

    def build(self, input_shape):

        # Add a certian number of layers
        for i in range(self.layers):
            self.model.add(Dense(self.nodes, input_shape=input_shape, activation='relu'))
            self.model.add(Dropout(self.drop_percent))

        # Output layer for the binary classification task
        self.model.add(Dense(output_shape=1, activation='sigmoid'))

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, batch_size, epochs, validation_data=None):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

# Load in the dataframe with the indicators and the embeddings
FE = "data (new implementation)/sentiment_exploration.parquet"
OD = Path.cwd()
DF_path = OD / FE
DF = pd.read_parquet(DF_path)

S_model = NeuralNetwork(1,100,0.20)


print('Loaded in something')