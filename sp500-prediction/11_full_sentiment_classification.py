# Classics
import keras.optimizers
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
from tqdm import tqdm

# Built-ins
from random import choice
from datetime import datetime
from dateutil.relativedelta import relativedelta

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

    def train(self, x_train, y_train, batch_size, epochs, CBs=None, validation_data=None,
              validation_percent=0, verbose=0):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_percent,
                                 callbacks=CBs, validation_data=validation_data, verbose=verbose,
                                 shuffle=False)
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

# Load in the dataframe with embeddings and the FF factors
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
combined_df.index.names = ['symbol', 'date']
combined_df.sort_index(level='date', inplace=True)
combined_column_names = combined_df.columns

# Hard code best model from the hyperparameter tuning
start_date = datetime(1999, 12, 31)

# Split the data into dependent and independent variable sets
X = combined_df[combined_column_names[15:]]

# Try the next day returns as the only dependent variable
dependent_df = combined_df['idosyncratic_change']

# Standard scalar
scaler1 = StandardScaler()

# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=False
)

sentiment_df_list = []
for i in tqdm(range(20)):

    # Calculate a time delta of a year and add it to the current start date
    end_date = start_date + relativedelta(years=3+i)
    predict_date = end_date + relativedelta(years=1)
    X_train = X.loc[(X.index.get_level_values('date') >= start_date) &
                    (X.index.get_level_values('date') <= end_date)]
    X_predict = X.loc[(X.index.get_level_values('date') > end_date) &
                    (X.index.get_level_values('date') <= predict_date)]

    y_train = dependent_df[X_train.index]

    # Classify articles that came out after positive day returns as "1", else "0"
    YC = np.where(y_train < 0, 0, 1).astype(int)

    # Scale the X data
    scaler = scaler1.fit(X_train)
    scaled_X_train = scaler.transform(X_train).astype(np.float32)
    scaled_X_predict = scaler.transform(X_predict).astype(np.float32)

    # Train the model on the current dataset
    model = NeuralNetwork()
    model.build(layers=1, npl=[836], LA=['relu'], drop_percent=[0], input_shape=scaled_X_train.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                       metrics=['accuracy'])
    hist = model.train(x_train=scaled_X_train, y_train=YC, epochs=50, batch_size=50, CBs=[early_stopping])

    # Predict the sentiment for the next year
    sentiment_vector = model.predict(scaled_X_predict)

    # Create a temporary sentiment DataFrame and add it to a list
    sentiment_df_temp = pd.DataFrame(sentiment_vector, columns=['sentiment'], index=X_predict.index)

    sentiment_df_list.append(sentiment_df_temp)

sentiment_all_df = pd.concat(sentiment_df_list, axis=0)

# Sort by symbol and date
sentiment_all_df.sort_index(inplace=True)

# Save the table
sentiment_save_path = Path("data/sentiment_all.parquet")
sentiment_all_df.to_parquet(sentiment_save_path)