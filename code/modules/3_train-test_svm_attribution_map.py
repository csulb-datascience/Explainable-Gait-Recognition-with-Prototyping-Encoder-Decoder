# csulb-datascience
#
# Module: train-test_svm_attribution_map
#   This module trains the OSVM classifier with the few-shot learning method, but
#   test it with embeddings obtained from occluded inputs.
#
#   It requires the predicted embeddings obtained when training the encoder-decoder,
#   and the attribution maps computed for each XAI method.
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

import tensorflow as tf
import numpy as np
import pandas as pd

import sys
sys.path.append("../classes")
from Recognition_Dataset_V1 import Dataset
from Recognition_EncoderModel_V2 import EncoderModel
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_SVM_V4 import SVM

#Creates the header for the file of results
def saveHeader(saveAs):
    #save the header
    values = [["iteration", "gamma", "nu", "tau", "lambda", "TPR", "TNR", "ACC", "TP", "FP", "FN", "TN", "method", "percent","position"]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a") 

#save the results of testing the OSVM 
def save(saveAs, iteration, gamma, nu, tau, lambdaVal, results, method, percent, position):
    #get values:
    tpr = results[0]
    tnr = results[1]    
    tp = results[2][2]
    fp = results[3][2] + results[3][3]
    fn = results[2][1] + results[2][3]
    tn = results[3][1]
    accuracy = (tp + tn) / (tp + fn + tn + fp)
                
    values = [[iteration, gamma, nu, tau, lambdaVal, tpr, tnr, accuracy, tp, fp, fn, tn, method, percent, position]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a")  

#Reads the attribution maps
def getAttributionMap(path, fileName, highRelevance=True):
    #transform the 2D data in a 1D list of pairs [value, (i,j)]
    heatMap2D = np.array(pd.read_csv(path + "/" + fileName))
    heatMap1D=[]
    for i in range(heatMap2D.shape[0]):
        for j in range(heatMap2D.shape[1]):
            heatMap1D.append([abs(heatMap2D[i,j]), (i,j)])

    #Also return the 1D heat map sorted according to relevance
    heatMap1D.sort(reverse = highRelevance)
    return(heatMap2D, heatMap1D)

#Get the occluding mask of some percentage
def getOccludingMask(shape, heatMap1D, percentage, initPoint=0):
    mask = np.ones(shape).astype("int")
    counter = (len(heatMap1D) * percentage) // 100

    #Select the mask accordi
    index=initPoint
    while counter>0 and index < len(heatMap1D):
        i,j = heatMap1D[index][1]
        mask[i,j]=0
        counter -= 1
        index += 1
            
    #Also return the last index that was not considered in the window
    return(mask, index)   


#CSV file of results
saveAs = "./svm_attribution_map_results.csv"
saveHeader(saveAs)    
    
#Location of files
mapsPath = "../XAI/"                 #folder that contains the attribution maps
filesPath = "./files/"               #folder of the embeddings and encoders
settingsPath = "../datasettings/"    #location of the data distribution settings
datasetPath = "../data/"             #location of the unit steps data
datasetFile = "dataset.csv"          #name of the csv file that contains the unit steps per participant

#load the data
print("--> loading the unit steps dataset.")
dataset = Dataset(datasetPath, datasetFile)

#parameters
lambdas = [1.0]                      #weight of the prototype loss
taus = [-0.1]                        #threshold for the classifier
gammas = [2.2]                       #hyperparameter to train the OSVM classifier
nus=[0.06]                           #hyperparameter to train the OSVM classifier
k_units = 10                         #number of units for the few-shot training set
methods = ["SA", "LRP"]              #XAI methods used.
percent = 20                         #percent that covers each occluding map
positions = 5                        #number of occluding maps
iterations=10                        #number of trainings

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

        #Load the encoder
        print ("loading the encoder")
        encoderModel = EncoderModel()
        encoder = encoderModel.loadModel(filesPath + "encoder_" + fileNameEnding + ".h5")    

        #load the embeddings
        print("loading embeddings")
        embeddings0 = EmbeddingsDataset(None, dataset)
        embeddings0.load(filesPath,  "embeddings_" + fileNameEnding + ".npy")

        #Train and test for each XAI method
        for method in methods:
            #load the attribution maps
            hmPath = mapsPath + "/" + method + "/" + str(lambdaVal) + "/"
            mapPress2D, mapPress1D = getAttributionMap(hmPath , "/Pressure/tot_emb_node_Pres_iter" + str(i) +".csv")
            mapAcc2D, mapAcc1D = getAttributionMap(hmPath , "/Acceleration/tot_emb_node_Acc_iter" + str(i) +".csv")
            mapRot2D, mapRot1D = getAttributionMap(hmPath , "/Rotation/tot_emb_node_Rot_iter" + str(i) +".csv")

            #get the occlusion map for each position
            indexPress, indexAcc, indexRot = 0, 0, 0
            for position in range(1, positions+1):
                #Create the mask
                maskPress, indexPress  = getOccludingMask(mapPress2D.shape, mapPress1D, percent, indexPress)
                maskAcc, indexAcc  = getOccludingMask(mapAcc2D.shape, mapAcc1D, percent, indexAcc)
                maskRot, indexRot  = getOccludingMask(mapRot2D.shape, mapRot1D, percent, indexRot)
                dataset.mask = np.concatenate((maskPress,maskAcc,maskRot), axis=1)

                #Apply the mask to the dataset and predict the embeddings 
                embeddingsMasked = EmbeddingsDataset(encoder, dataset)
                embeddingsMasked.predictEmbeddings(masking=True)

                #Train SVM with different parameters
                print("--> SVM Training and testing")
                for gamma in gammas:
                    for nu in nus:
                        print("--> Train SVM: gamma=", gamma, " nu=", nu)
                        svm = SVM(embeddings0)                             #assign the embeddings for OSVM training
                        svm.fit(randomTrainingSet, gamma=gamma, nu = nu)   #train the classifier with k units
                        svm.embeddings = embeddingsMasked                  #assign the occluded embeddings for testing

                        #Test the classifier for different thresholds using both known-test and unknown-test sets
                        #Save the results in the specified file
                        for tau in taus:                        
                            results = svm.accuracy(randomKnownSet, dataset.unseenSet, tau)
                            save(saveAs, i, gamma, nu, tau, lambdaVal, results, method, percent, position)   
