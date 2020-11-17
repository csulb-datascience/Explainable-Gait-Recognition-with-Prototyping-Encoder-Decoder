"""
   This module trains the autoencoder, saves the trained encoder and the predicted embeddings

"""
import tensorflow as tf
import numpy as np
import pandas as pd
import time

import sys
sys.path.append("../classes")
from Recognition_Autoencoder_V2 import Autoencoder
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_Dataset_V1 import Dataset

#parameters
numberPeopleTraining = 16
numberPeopleKnown = 7
batchSize = 64
learningRate = 0.001
alpha = 1.25 
beta = 1.0
lambdas = [0.0, 0.5, 1.0, 2.0, 3.0]
epochs = 20
iterations=10
storePath = "./files/"
settingsPath = "../datasettings_new/"

#load the data
print("--> loading the data")
dataset = Dataset("../data", "dataset.csv")

#Iterations
for i in range(1, iterations+1):
    print("\n--> Iteration: ", i, " Alpha: ", alpha)

    #prepare the dataset indexes: training, validation, unseen, test
    print("--> Getting the datasettings")
    dataset.loadSets(settingsPath, "dataset_iter_" + str(i) + ".npy")

    x_train, y_train, m_train = dataset.getDataset(dataset.trainingSet, batchSize=batchSize)
    print("--> dataset ready.")

    for j, lambdaVal in enumerate(lambdas):        
        print("\n--> Iteration: ", i, " lambda: ", lambdaVal)

        # Creates a session 
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)            
        with session.as_default():   
                 
            print("--> Model ")
            autoencoder = Autoencoder(dataset.unitSize(), alpha, beta, lambdaVal, learningRate)
            outerModel, encoder = autoencoder.getCompiledCNN()
            print("\n\n--> Training ")
            history=outerModel.fit(x_train, y={"encoder": y_train, "decoder": m_train}, batch_size = batchSize, epochs = epochs)

            #Save the encoder
            print("\n--> saving the encoder and embeddings")
            encoder.save(storePath + "encoder_iter_" + str(i) + "_lambda_" + str(lambdaVal) + ".h5")

            #Save embeddings
            embeddings = EmbeddingsDataset(encoder, dataset)
            embeddings.predictEmbeddings()
            embeddings.save(storePath, "embeddings_iter_" + str(i) + "_lambda_" + str(lambdaVal) +  ".npy")

        tf.compat.v1.reset_default_graph()
        session.close()
 
       