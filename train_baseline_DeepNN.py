# train_baseline_DeepNN.py
###########################################################
# Trains and saves baseline's deep neural network model
###########################################################
# Expects 2 command line arguments
# arg1 - filepath to input data (CSV file)
# arg2 - filepath to user labels

from tensorflow import keras
from tensorflow.keras import layers
from numpy import loadtxt
import time
import sys

train_set = loadtxt(sys.argv[1], delimiter=',')
user_labels = loadtxt(sys.argv[2], delimiter=',')

user_features = 2
game_features = 6
user_game_features = 2
num_games = 3000

layer_size_input = user_features + ((game_features + user_game_features) * num_games)
layer_size_hidden_1 = 128
layer_size_hidden_2 = 128
layer_size_output = 1

input = keras.Input(shape=(layer_size_input,))

hidden_1 = layers.Dense(layer_size_hidden_1, activation='relu', name='hidden_1')(input)
hidden_2 = layers.Dense(layer_size_hidden_2, activation='relu', name='hidden_2')(hidden_1)

output = layers.Dense(layer_size_output, activation='sigmoid', name='output')(hidden_2)

model = keras.Model(inputs=input, outputs=output)

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss="mean_squared_error")
model.fit(train_set, user_labels, epochs=100)

timestr = time.strftime("%Y%m%d-%H%M%S")
model.save("model_baseline_DeepNN_"+timestr+".h5")