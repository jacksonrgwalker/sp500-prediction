# Load in packages

# Classics
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning
import tensorflow as tf
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Built-ins
from random import choice

class NeuralNetwork():
    def __init__(self):
        self.model = Sequential()

    def build(self, layers, npl, LA, drop_percent, input_shape):

        # Add a certian number of layers
        for i in range(layers):
            if i == 0:
                self.model.add(Dense(npl[i], activation=LA[i], input_dim=input_shape))
            else:
                self.model.add(Dense(npl[i], activation=LA[i]))

            self.model.add(Dropout(rate=drop_percent[i]))

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
if tf.test.gpu_device_name():
    print("GPU is available, but turning it off for this network")
    tf.config.set_visible_devices([], 'GPU')
else:
    print("GPU is NOT available, continuing with CPU")

# Load in the dataframe with OHLC data as well as Fama-French 5-Factor data
operating_directory = Path.cwd()
embedding_file_extension = "data/sentiment_exploration (1).parquet"
beta_file_extension = "data/beta.parquet"

# Aggregate news to a single embedding per day, per ticker, by averaging embeddings if
# the same ticker has multiple articles for a single day
embedding_df = pd.read_parquet(operating_directory / embedding_file_extension)
column_names = embedding_df.columns
embedding_df = embedding_df.drop(column_names[0], axis=1).groupby([column_names[1],
                                column_names[2]]).mean().dropna()


beta_df = pd.read_parquet(operating_directory / beta_file_extension)

# Join on both multi-indices
multi_index_names = embedding_df.index.names
combined_df = embedding_df.join(beta_df, on=multi_index_names)
combined_df.dropna(inplace=True)
combined_column_names = combined_df.columns

# Split the data into dependent and independent variable sets
X = combined_df.loc[:, combined_column_names[15:]].to_numpy()

# Try the next day returns as the only dependent variable
dependent_df = combined_df['idosyncratic_change']

# Classify articles that came out after positive day returns as "1", else "0"
YC = np.where(dependent_df < 0, 0, 1)

# Scale the data with a standard scaler
scaler1 = StandardScaler()

train = 0

if train == 0:

    # Load in the model
    best_model = load_model("data/best_model.h5")

else:

    # Split the data into train and test
    test_fraction = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, YC, test_size=test_fraction)

    # Tensors for tensorflow
    scaled_X_train = scaler1.fit_transform(X_train).astype(np.float32)
    scaled_X_test = scaler1.fit_transform(X_test).astype(np.float32)

    # Define a list of hyperparameters to test
    dummy = []
    feature_size = scaled_X_train.shape[1]
    HP = {}
    HP['Optimizer'] = ['SGD', 'Adam']
    HP['HLs'] = [1, 2]
    HP['NPL'] = [[int(feature_size*2/3) - 64*i for i in range(10)],
                 [32*i for i in range(1, 11)]]
    HP['DR'] = [0.0, 0.10, 0.20]
    HP['batch_size'] = [50, 100, 150]
    HP['AF'] = ['relu', 'tanh']

    # Try 50 different combinations of the hyperparameters and see if we can get something that fits well
    combs = 3
    cc = 1
    SCs = []
    epochs = 50
    MVA = 0

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=False
    )

    for i in range(combs):

        # Choose the hyperparameters randomly and append the list to a dictionary
        CPs = {}
        CPs['CO'] = choice(HP['Optimizer']) # Optimizer
        CPs['CBS'] = choice(HP['batch_size']) # Batch size
        CPs['CLs'] = choice(HP['HLs'])  # Layers
        CPs['CNPLs'] = [choice(HP['NPL'][i]) for i in range(CPs['CLs'])]   # Nodes per layer
        CPs['CDRs'] = [choice(HP['DR']) for i in range(CPs['CLs'])]     # Dropout per layer
        CPs['CAs'] = [choice(HP['AF']) for i in range(CPs['CLs'])]      # Activation function per layer

        attempts = 0
        while CPs in SCs and attempts < 10:
            CPs['CLs'] = choice(HP['HLs'])
            CPs['CNPLs'] = [choice(HP['NPL'][i]) for i in range(CPs['CLs'])]
            CPs['CDRs'] = [choice(HP['DR']) for i in range(CPs['CLs'])]
            CPs['CAs'] = [choice(HP['AF']) for i in range(CPs['CLs'])]
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
        model.build(CPs['CLs'], CPs['CNPLs'], CPs['CAs'], CPs['CDRs'], scaled_X_train.shape[1])
        model.compile(CPs['CO'], loss='binary_crossentropy')
        hist = model.train(x_train=scaled_X_train, y_train=y_train, epochs=epochs, batch_size=CPs['CBS'],
                           validation_data=(scaled_X_test, y_test), CBs=[early_stopping])

        if np.max(hist.history['val_accuracy']) > MVA:
            Best_params = CPs
            MVA = np.max(hist.history['val_accuracy'])

            if MVA > 0.61:
                break

        print('=====================================================================')
        print('')


    best_model = NeuralNetwork()
    best_model.build(Best_params['CLs'], Best_params['CNPLs'], Best_params['CAs'], Best_params['CDRs'],
                     scaled_X_train.shape[1])
    best_model.compile(Best_params['CO'], loss='binary_crossentropy')
    hist = best_model.train(x_train=scaled_X_train, y_train=y_train, epochs=epochs, batch_size=CPs['CBS'],
                       validation_data=(scaled_X_test, y_test))

    best_model.model.save('data/best_model')

# With the best model determined, predict the labels for the dataset
scaled_X = scaler1.fit_transform(X).astype(np.float32)
labels = best_model.predict(scaled_X)

# Create a DataFrame to store the sentiment
sentiment_df = pd.DataFrame(labels, columns=['sentiment_score'], index=combined_df.index)

# Save the table
sentiment_save_path = Path("data/sentiment.parquet")
sentiment_df.to_parquet(sentiment_save_path)