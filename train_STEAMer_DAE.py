# train_STEAMer_DAE.py
###########################################################
# Trains and saves STEAMer's deep autoencoder model
###########################################################
# Expects 1 command line arguments
# arg1 - filepath to input data (CSV file)

from tensorflow import keras
from tensorflow.keras import layers
from numpy import loadtxt
import time
import sys

train_set = loadtxt(sys.argv[1], delimiter=',')

user_features = 2
game_features = 6
user_game_features = 3
num_games = 3000

layer_size_input = user_features + ((game_features + user_game_features) * num_games)
layer_size_hidden_1 = 256
layer_size_hidden_2 = 128
layer_size_hidden_3 = 256
layer_size_output = layer_size_input

input = keras.Input(shape=(layer_size_input,))

hidden_1 = layers.Dense(layer_size_hidden_1, activation='relu', name='hidden_1')(input)
hidden_2 = layers.Dense(layer_size_hidden_2, activation='relu', name='hidden_2')(hidden_1)
hidden_3 = layers.Dense(layer_size_hidden_3, activation='relu', name='hidden_3')(hidden_2)

output = layers.Dense(layer_size_output, activation='sigmoid', name='output')(hidden_3)

model = keras.Model(inputs=input, outputs=output)

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss="mean_absolute_error")
model.fit(train_set, train_set, epochs=20)

timestr = time.strftime("%Y%m%d-%H%M%S")
model.save("model_STEAMer_DAE_"+timestr+".h5")