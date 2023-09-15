# Load in packages
import tensorflow as tf
import keras
import keras.optimizers
import numpy as np
import pandas as pd
from pathlib import Path
from keras import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras_tuner.tuners import RandomSearch
from copy import deepcopy as dc

# Create a neural net function to train the hyperparameters
def build_model(hp):
    model = keras.Sequential()
    model.add(Input(shape=(SX_train_tens.shape[1],)))  # Specify the input size here

    # Tune the number of layers
    hidden_layers = hp.Int('hidden_layers', min_value=2, max_value=4, step=1)

    # Tune the number of neurons per layer
    for i in range(hidden_layers):
        units = hp.Int(f'units_{i+1}', min_value=32, max_value=512, step=32)
        activation_H = hp.Choice(f'activation_{i+1}', ['relu', 'tanh'])
        dropout_rate = hp.Float(f'dropout_{i+1}', min_value=0.0, max_value=0.5, step=0.1)

        # Add dropout layer and tune the dropout rate
        model.add(Dense(units=units, activation=activation_H))
        model.add(Dropout(rate=dropout_rate))

    # Output layer with a single neuron and sigmoid activation for binary classification
    model.add(Dense(1, activation='linear'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    # Tune the batch size
    batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model

class NeuralNetwork():
    def __init__(self):
        self.model = Sequential()

    def build(self, layers, npl, input_shape, drop_percent):

        # Add a certian number of layers
        for i in range(layers):
            if i == 0:
                self.model.add(Dense(npl[i], input_shape=(input_shape, ), activation='relu'))
            else:
                self.model.add(Dense(npl[i], activation='relu'))

            self.model.add(Dropout(drop_percent))

        # Output layer for the binary classification task
        self.model.add(Dense(1, activation='sigmoid'))

    def compile(self, optimizer, loss, metrics=['mean_squared_error']):
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

# Aggregate news to a single embedding per day, per ticker, by averaging embeddings if
# the same ticker has multiple articles for a single day
CNs = DF.columns
DF_Agg = DF.drop(CNs[0], axis=1).groupby([CNs[2], CNs[1]]).mean().dropna()
ACNs = DF_Agg.columns

# Split the data into dependent and independent variable sets
X = DF_Agg.loc[:, ACNs[15:]].to_numpy()

# Try the next day returns as the only dependent variable
Returns_DF = DF_Agg.loc[:, ACNs[0:14]]
# Y = Returns_DF.iloc[:, 2]
Y = Returns_DF.iloc[:, 2]

# Apply the custom function to create the new column
# YC = np.where(Y < 0, 0, 1)
YC = dc(Y).to_numpy()

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, YC, test_size=0.05)

# Scale the data with a standard scaler
scaler1 = MinMaxScaler(feature_range=(-1,1))
scaler2 = StandardScaler()

# Tensors for tensorflow
SX_train_tens = tf.convert_to_tensor(scaler2.fit_transform(X_train).astype(np.float32))
y_train_tens = tf.convert_to_tensor(scaler1.fit_transform(y_train.reshape(-1, 1)).astype(np.float32))
SX_test_tens = tf.convert_to_tensor(scaler2.fit_transform(X_test).astype(np.float32))
y_test_tens = tf.convert_to_tensor(scaler1.fit_transform(y_test.reshape(-1, 1)).astype(np.float32))

# Search for optimal hyperparameters
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=1)

# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,            # Number of epochs with no improvement before stopping
    restore_best_weights=False  # Restore the best model weights
)

tuner.search(SX_train_tens, y_train_tens, epochs=100,
             validation_data=(SX_test_tens, y_test_tens), callbacks=[early_stopping])

# Print out stats on the best model
print(tuner.results_summary())

# best_models = tuner.get_best_models(num_models=1)
# test_loss, test_acc = best_models[0].evaluate(SX_test_tens, y_test_tens)
# print('Test accuracy:', test_acc)

