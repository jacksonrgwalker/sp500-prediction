# Load in packages

# Classics
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning
import tensorflow as tf
import keras
import keras.optimizers
from keras import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Built-ins
from copy import deepcopy as dc
from random import choice

# Create a neural net function to train the hyperparameters
def build_model(hp):
    model = keras.Sequential()
    model.add(Dense(shape=(SX_train.shape[1],)))  # Specify the input size here

    # Tune the number of neurons per layer
    for i in range(hp.Int('layers', min_value=1, max_value=3)):

        # Add dropout layer and tune the dropout rate
        model.add(Dense(units=hp.Int(f'units_{i+1}', min_value=32, max_value=512, step=32),
                        activation=hp.Choice(f'activation_{i+1}', ['relu', 'tanh'])))
        model.add(Dropout(rate=hp.Float(f'dropout_{i+1}',
                                        min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer with a single neuron and sigmoid activation for binary classification
    model.add(Dense(1, activation='linear'))

    # Tune the batch size
    batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

    # Compile the model
    model.compile(optimizer=hp.Choice('Optimizer', values=['SGD', 'adam']),
        loss='mean_squared_error',
        metrics=['mean_squared_error'])

    return model

class NeuralNetwork():
    def __init__(self):
        self.model = Sequential()

    def build(self, layers, npl, LA, drop_percent, input_shape):

        # Add input layer
        self.model.add(Dense(units=input_shape, input_shape=(input_shape,), activation = 'relu'))

        # Add a certian number of layers
        for i in range(1, layers):
            self.model.add(Dense(npl[i-1], activation=LA[i-1]))
            self.model.add(Dropout(rate=drop_percent[i-1]))

        # Output layer for the binary classification task
        self.model.add(Dense(1, activation='sigmoid'))

    def compile(self, optimizer, loss, metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, batch_size, epochs, CBs=None, validation_data=None, verbose=1):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 callbacks=CBs, validation_data=validation_data, verbose=verbose)
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

# Check if GPU is available for GPU
# if tf.test.gpu_device_name():
#     print("GPU is available, but turning it off for this network")
#     tf.config.set_visible_devices([], 'GPU')
# else:
#     print("GPU is NOT available, continuing with CPU")

# Load in the dataframe with OHLC data as well as Fama-French 5-Factor data
operating_directory = Path.cwd()
embedding_file_extension = "data/sentiment_exploration (1).parquet"
beta_file_extension = "data/beta.parquet"

embedding_df = pd.read_parquet(operating_directory / embedding_file_extension)
column_names = embedding_df.columns
embedding_df = embedding_df.drop(column_names[0], axis=1).groupby([column_names[1],
                                column_names[2]]).mean().dropna()

beta_df = pd.read_parquet(operating_directory / beta_file_extension)

# Join on both multi-indices
multi_index_names = embedding_df.index.names
combined_df = embedding_df.join(beta_df, on=multi_index_names)
combined_df.dropna(inplace=True)

# Aggregate news to a single embedding per day, per ticker, by averaging embeddings if
# the same ticker has multiple articles for a single day
combined_column_names = combined_df.columns

# Split the data into dependent and independent variable sets
X = combined_df.loc[:, combined_column_names[15:]].to_numpy()

# Try the next day returns as the only dependent variable
dependent_df = combined_df['idosynchratic_change']

# Apply the custom function to create the new column
YC = np.where(dependent_df < 0, 0, 1)
# YC = dc(Y).to_numpy()

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, YC, test_size=0.20)

# Scale the data with a standard scaler
scaler1 = MinMaxScaler(feature_range=(-1,1))
scaler2 = StandardScaler()

# Tensors for tensorflow
SX_train = scaler2.fit_transform(X_train).astype(np.float32)
SX_test = scaler2.fit_transform(X_test).astype(np.float32)

# Define a list of hyperparameters to test
HP = {
    'Optimizer': ['SGD', 'Adam'],#, 'RMSprop', 'Nadam', 'Adagrad', 'Adadelta'],
    'HLs': [1],
    'NPL': [int(i*32) for i in range(1,31)],
    'DR': [0.0, 0.10, 0.20],
    'batch_size': [int(i*10) for i in range(1,11)],
    'AF': ['relu', 'tanh', 'sigmoid']
}

# Try 1000 different combinations of the hyperparameters and see if we can get something that fits well
combs = 500
cc = 1
SCs = []
epochs = 25
MVA = 0

accuracy_ = tf.keras.metrics.BinaryAccuracy(
    name='binary_accuracy', dtype=None, threshold=0.5
)

# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_'+accuracy_.name,
    patience=5,
    restore_best_weights=False
)

for i in range(combs):

    # Choose the hyperparameters randomly and append the list to a dictionary
    CPs = {}
    CPs['CO'] = choice(HP['Optimizer']) # Optimizer
    CPs['CBS'] = choice(HP['batch_size']) # Batch size
    CPs['CLs'] = choice(HP['HLs'])  # Layers
    CPs['CNPLs'] = [choice(HP['NPL']) for i in range(CPs['CLs']-1)]   # Nodes per layer
    CPs['CDRs'] = [choice(HP['DR']) for i in range(CPs['CLs']-1)]     # Dropout per layer
    CPs['CAs'] = [choice(HP['AF']) for i in range(CPs['CLs']-1)]      # Activation function per layer

    attempts = 0
    while CPs in SCs and attempts < 10:
        CPs['CLs'] = choice(HP['HLs'])
        CPs['CNPLs'] = [choice(HP['NPL']) for i in range(CPs['CLs']-1)]
        CPs['CDRs'] = [choice(HP['DR']) for i in range(CPs['CLs']-1)]
        CPs['CAs'] = [choice(HP['AF']) for i in range(CPs['CLs']-1)]
        attempts += 1

    # Couldn't find a new combination to try
    if attempts == 10:
        break

    # Print out the combination and the iteration we're on then store the current combination
    print(f'The best validation accuracy thus far is {MVA}')
    print('')
    print(f'Trying combination {i}: {CPs}')
    print('')
    SCs.append(CPs)

    model = NeuralNetwork()
    model.build(CPs['CLs'], CPs['CNPLs'], CPs['CAs'], CPs['CDRs'], SX_train.shape[1])
    model.compile(CPs['CO'], loss='binary_crossentropy', metrics=[accuracy_])
    hist = model.train(x_train=SX_train, y_train=y_train, epochs=epochs, batch_size=CPs['CBS'],
                       validation_data=(SX_test, y_test), CBs=[early_stopping])

    if np.max(hist.history['val_'+accuracy_.name]) > MVA:
        Best_params = CPs
        MVA = np.max(hist.history['val_'+accuracy_.name])

    print('=====================================================================')
    print('')

best_model = NeuralNetwork()
best_model.build(Best_params['CLs'], Best_params['CNPLs'], Best_params['CAs'], Best_params['CDRs'], SX_train.shape[1])
best_model.compile(CPs['CO'], loss='binary_crossentropy', metrics=['accuracy'])
best_model.save('Best_model.H5')
