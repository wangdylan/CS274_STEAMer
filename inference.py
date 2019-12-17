# inference.py
###########################################################
# Loads a keras model and runs a prediction on input data
# Saves prediction results as a CSV file
###########################################################
# Expects 2 command line arguments
# arg1 - filepath to input data
# arg2 - filepath to saved model

from tensorflow import keras
from numpy import loadtxt
from numpy import savetxt
import sys
import time

data = loadtxt(sys.argv[1], delimiter=',')

model = keras.models.load_model(sys.argv[2])
model.summary()
result = model.predict(data)

timestr = time.strftime("%Y%m%d-%H%M%S")
savetxt("output_" + timestr + ".csv", result, delimiter=",")
