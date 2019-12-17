# CS274_STEAMer
Course project by Dylan Wang for CS 274 - Fall 2019 at SJSU.
-
STEAMer - Prototype game recommendation system for the Steam platform using a deep autoencoder and Steam's user data.
-
Run the training Python scripts with TensorFlow to produce and save a model for that specific setup.
Necessary command line arguments detailed in source code files.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train_baseline_DeepNN.py - Produces a deep neural network based model that is used as baseline
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train_STEAMer_DeepNN.py - Produces a deep neural network based model like baseline but with newer feature set
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train_STEAMer_DAE - Produces a deep autoencoder model with newer feature set

Run inference.py with input data and a saved model for predictions
*************************************************************************************************************************************

Dataset files are too large to be uploaded to GitHub.
They can be found from the original source as a SQL dump at: https://steam.internet.byu.edu/
