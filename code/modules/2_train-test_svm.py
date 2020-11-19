# csulb-datascience
#
# Module: train-test_svm
#   This module trains and test the OSVM classifier
#   It requires the predicted embeddings obtained when training the encoder-decoder.
#   The results are saved in a CSV file for later plot and analysis.
#
# File name format of the embeddings file:
#   Embeddings:  "embeddings_iter_" + iteration + "_lambda_" + lambda value + ".npy" 
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Sella Bae,     email: sella.bae@student.csulb.edu
#
# Date: Nov 2020
#
# Include a reference to this site if you will use this code.

import numpy as np
import pandas as pd

import sys
sys.path.append("../classes")
from Recognition_Dataset_V1 import Dataset
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_SVM_V4 import SVM

#Creates the header for the file of results
def saveHeader(saveAs):
    #save the header
    values = [["iteration", "gamma", "nu", "tau", "lambda", "TPR", "TNR", "ACC", "TP", "FP", "FN", "TN"]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a") 

#save the results of testing the OSVM 
def save(saveAs, iteration, gamma, nu, tau, lambdaVal, results):
    #get values:
    tpr = results[0]
    tnr = results[1]    
    tp = results[2][2]
    fp = results[3][2] + results[3][3]
    fn = results[2][1] + results[2][3]
    tn = results[3][1]
    accuracy = (tp + tn) / (tp + fn + tn + fp)
                
    values = [[iteration, gamma, nu, tau, lambdaVal, tpr, tnr, accuracy, tp, fp, fn, tn]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a")  


#CSV file of results
saveAs = "./svm_results.csv"
saveHeader(saveAs)    
    
#Location of files
embeddingsPath = "./files/"          #folder of the embeddings
settingsPath = "../datasettings/"    #location of the data distribution settings
datasetPath = "../data/"             #location of the unit steps data
datasetFile = "dataset.csv"          #name of the csv file that contains the unit steps per participant

#load the data
print("--> loading the unit steps dataset.")
dataset = Dataset(datasetPath, datasetFile)

#parameters
lambdas = [0.0, 1.0, 2.0, 3.0]              #weight of the prototype loss
taus = np.linspace(0.0, 0.2, 21) * -1       #threshold for the classifier
gammas = np.linspace(0.1, 5, 50)            #hyperparameter to train the OSVM classifier
nus=np.round(np.linspace(0.01, 0.2, 20),2)  #hyperparameter to train the OSVM classifier
k_units = 10                                #number of units for the few-shot training set
iterations=10                               #number of trainings

#Repeat training 
for i in range(1, iterations+1):
    #Load the dataset settings
    dataset.loadSets(settingsPath, "dataset_iter_"  + str(i) + ".npy")

    #Get K random unit steps from the Known-Test set to train the OSVM, keep the remainder for testing
    randomTrainingSet, randomKnownSet = dataset.selectRandomUnits(dataset.validationSet, k_units)

    #Train for each required lambda
    for j, lambdaVal in enumerate(lambdas): 
        print("\n-->Iteration: ", i, " Lamda: ", lambdaVal)

        #Use the same name endings of files created.
        fileNameEnding = "iter_" + str(i) + "_lambda_" + str(lambdaVal) 

        #load the embeddings
        print("loading embeddings")
        embeddings0 = EmbeddingsDataset(None, dataset)
        embeddings0.load(embeddingsPath,  "embeddings_" + fileNameEnding + ".npy")

        #Train SVM with different parameters
        print("--> SVM Training and testing")
        for gamma in gammas:
            for nu in nus:
                print("--> Train SVM: gamma=", gamma, " nu=", nu)
                svm = SVM(embeddings0)
                svm.fit(randomTrainingSet, gamma=gamma, nu = nu)   #train the classifier with k units

                #Test the classifier for different thresholds using both known-test and unknown-test sets
                #Save the results in the specified file
                for tau in taus:                        
                    results = svm.accuracy(randomKnownSet, dataset.unseenSet, tau)
                    save(saveAs, i, gamma, nu, tau, lambdaVal, results)   
