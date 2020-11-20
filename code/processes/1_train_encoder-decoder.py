# csulb-datascience
#
# Process: train_encoder-decoder
#   This process trains the encoder-decoder model.
#   It saves the trained encoder and the predicted embeddings for later processes.
#
# File name format:
#   Encoder:     "encoder_iter_" + iteration + "_lambda_" + lambda value + ".h5" 
#   Embeddings:  "embeddings_iter_" + iteration + "_lambda_" + lambda value + ".npy" 
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Sella Bae,     email: sella.bae@student.csulb.edu
#
# Date: Nov 2020
#
# Include a reference to this site if you will use this code.

import tensorflow as tf
import numpy as np
import pandas as pd
import time

import sys
sys.path.append("../classes")
from Recognition_Autoencoder_V2 import Autoencoder
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_Dataset_V1 import Dataset

#Location of files
storePath = "./files/"               #folder to save the resultant models and embeddings
settingsPath = "../datasettings/"    #location of the data distribution settings
datasetPath = "../data/"             #location of the unit steps data
datasetFile = "dataset.csv"          #name of the csv file that contains the unit steps per participant

#load the data
print("--> loading the unit steps dataset.")
dataset = Dataset(datasetPath, datasetFile)

#parameters
batchSize = 64                          #batch size for training
learningRate = 0.001                    #learning rate for training
alpha = 1.25                            #hyperparameter "margin" in the Triplet Loss
lambdas = [0.0, 1.0, 2.0, 3.0]          #weight of the prototype loss
epochs = 20                             #number of epochs for training
iterations=10                           #number of trainings

#Iterations
for i in range(1, iterations+1):
    print("\n--> Iteration: ", i, " Alpha: ", alpha)

    #prepare the training dataset
    print("--> Getting the data settings")
    dataset.loadSets(settingsPath, "dataset_iter_" + str(i) + ".npy")
    x_train, y_train, m_train = dataset.getDataset(dataset.trainingSet, batchSize=batchSize)
    print("--> training set ready.")

    #Train for each required lambda
    for j, lambdaVal in enumerate(lambdas):        
        print("\n--> Iteration: ", i, " lambda: ", lambdaVal)

        # Creates a session 
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)            
        with session.as_default():   

            #Train an instance of de encoder-decoder with the defined parameters
            print("\n\n--> Training the encoder-decoder")
            autoencoder = Autoencoder(dataset.unitSize(), alpha=alpha, lambdaVal=lambdaVal, learningRate=learningRate)
            outerModel, encoder = autoencoder.getCompiledCNN()
            history=outerModel.fit(x_train, y={"encoder": y_train, "decoder": m_train}, batch_size= batchSize, epochs= epochs)

            #Use the same name endings for files created.
            fileNameEnding = "iter_" + str(i) + "_lambda_" + str(lambdaVal) 

            #Save the encoder
            print("\n--> saving the encoder")
            encoder.save(storePath + "encoder_" + fileNameEnding  + ".h5")

            #Save embeddings
            print("\n--> predicting and saving embeddings")
            embeddings = EmbeddingsDataset(encoder, dataset)
            embeddings.predictEmbeddings()
            embeddings.save(storePath, "embeddings_" + fileNameEnding + ".npy")

        #Reset the graph and close the session to clear the GPU memory
        tf.compat.v1.reset_default_graph()
        session.close()
 
       