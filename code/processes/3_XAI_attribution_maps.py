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
#  Authors:
#   Yong-Min Shin < jordan3414@yonsei.ac.kr>
#   Jin-Duk Park < jindeok6@yonsei.ac.kr>
# Date: Nov 2020
#
# Include a reference to this site if you will use this code.

import tensorflow as tf
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append("../classes")
from Recognition_EncoderModel_V2 import EncoderModel
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_Dataset_V1 import Dataset

# XAI part
import pandas as pd
import innvestigate.utils as iutils
import innvestigate
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def cumulate_analysis(idxlist, analyzed_np, feature = 0):
    '''
    feature = 0,1,2 (pres,acc,rot each)
    '''
    cumulate_np = np.zeros((np.shape(analyzed_np[feature][0])))
    for i in idxlist:
        cumulate_np += analyzed_np[feature][i]
    cumulate_np = cumulate_np / len(idxlist) # takes average

    return cumulate_np

def createLogFile(basePATH, filename):
    Path('\\'.join([basePATH, filename])).mkdir(exist_ok = True)
    
def createLogFile_all():
    mode_list = ["gradient", "lrp"]
    lambdas = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
    for mode in mode_list:
        for lamdval in lambdas:            
            createLogFile(os.getcwd(), '{}'.format(mode))    
            createLogFile(os.getcwd(), '{}\\{}'.format(mode,lamdval))
            createLogFile(os.getcwd(), '{}\\{}'.format(mode,lamdval))
            createLogFile(os.getcwd(), '{}\\{}'.format(mode,lamdval))
            createLogFile(os.getcwd(), '{}\\{}\\Acceleration'.format(mode,lamdval))
            createLogFile(os.getcwd(), '{}\\{}\\Pressure'.format(mode,lamdval))
            createLogFile(os.getcwd(), '{}\\{}\\Rotation'.format(mode,lamdval))

createLogFile_all()     

#Location of files
storePath = "./files/"               #folder to save the resultant models and embeddings
settingsPath = "../datasettings/"    #location of the data distribution settings
datasetPath = "../data/"             #location of the unit steps data
datasetFile = "dataset.csv"          #name of the csv file that contains the unit steps per participant

#load the data
print("--> loading the unit steps dataset.")
dataset = Dataset(datasetPath, datasetFile)

#parameters
lambdas = [1.0]
modes = ["gradient", "lrp"]
iterations=10

# XAI Mode config
XAI_each_latent = False

for mode in modes:      
    #Iterations
    for it in range(1, iterations+1):
        print("\n--> Iteration: ", it)    
        #Load the datasets
        dataset.loadSets(settingsPath, "dataset_iter_" + str(it) + ".npy")
    
        for j, lambdaVal in enumerate(lambdas): 
            print("\n-->Iteration: ", it, " Lamda: ", lambdaVal)
            posfix = "iter_" + str(it) + "_lambda_" + str(lambdaVal) 
    
            # Creates a session 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            with session.as_default():  
                              
                #Load the encoder
                print ("loading the encoder")
                encoderModel = EncoderModel()
                encoder = encoderModel.loadModel(storePath + "encoder_" + posfix + ".h5")    
                #Load the data
                x_train, y_train, m_train = dataset.getDataset(dataset.trainingSet, shuffled=False)

                print("----------- BEGIN XAI part ---------------")
                           
                analyzer = innvestigate.create_analyzer(mode, encoder,neuron_selection_mode="index", allow_lambda_layers = True)
                # Average users heatmap
                # Get indices of training users
                trainingPeople = np.array(list(dataset.trainingPeople()), dtype = int) - 1
                # divide & list up [user id-step id]
                stepidx_list = []
                
                for tt in range(1,33):
                
                    stepidx = np.where(y_train == tt)
                    stepidx_row = np.array(stepidx) #shape of (1,~~~)
                    stepidx_list.append(stepidx_row)
            
                user_analysis_list_P = []
                user_analysis_list_A = []
                user_analysis_list_R = []
    
                # List containing all heatmaps
                log_P = []
                log_A = []
                log_R = []
                
                # For every latent dimension...
                for emb_node_idx in tqdm(range(128), desc = 'Heatmap, avg all train users'):
                    # Get analysis of current dimension
                    analyzed = analyzer.analyze(x_train, emb_node_idx)

                    # cumulate analysis & takes average on it
                    for i in range(32):
                
                        idx_list_eachuser = stepidx_list[i][0].tolist()
                        
                        if mode == 'gradient':
                            avg_P = np.abs(cumulate_analysis(idx_list_eachuser, analyzed, feature = 0))
                            avg_A = np.abs(cumulate_analysis(idx_list_eachuser, analyzed, feature = 1))
                            avg_R = np.abs(cumulate_analysis(idx_list_eachuser, analyzed, feature = 2))
                        else:
                            avg_P = cumulate_analysis(idx_list_eachuser, analyzed, feature = 0)
                            avg_A = cumulate_analysis(idx_list_eachuser, analyzed, feature = 1)
                            avg_R = cumulate_analysis(idx_list_eachuser, analyzed, feature = 2)
                        user_analysis_list_P.append(avg_P)
                        user_analysis_list_A.append(avg_A)
                        user_analysis_list_R.append(avg_R)
                
                    # Get heatmap of only the training users
                    train_user_analysis_list_P = np.array(user_analysis_list_P)[trainingPeople]
                    train_user_analysis_list_A = np.array(user_analysis_list_A)[trainingPeople]
                    train_user_analysis_list_R = np.array(user_analysis_list_R)[trainingPeople]
                
                    # Take element-wise average of the collected heatmaps
                    train_user_analysis_mean_P = np.mean(train_user_analysis_list_P, axis = 0)
                    train_user_analysis_mean_A = np.mean(train_user_analysis_list_A, axis = 0)
                    train_user_analysis_mean_R = np.mean(train_user_analysis_list_R, axis = 0)
                
                    # Log heatmaps
                    log_P.append(train_user_analysis_mean_P)
                    log_A.append(train_user_analysis_mean_A)
                    log_R.append(train_user_analysis_mean_R)
                    
                    if XAI_each_latent == True:
                
                        # Plot & Output to excel
                        df_P = pd.DataFrame(train_user_analysis_mean_P)
                        df_A = pd.DataFrame(train_user_analysis_mean_A)
                        df_R = pd.DataFrame(train_user_analysis_mean_R)
                    
                        if mode == 'gradient':
                            sns.heatmap(df_P, annot=False, vmin = 0, vmax = 0.0055) # Fix y range
                        if mode == 'lrp.epsilon':
                            sns.heatmap(df_P, annot=False, vmin = -0.00015, vmax = 0.00015) # Fix y range
                        plt.title('{}(Pres), emb_node{}'.format(mode,emb_node_idx), fontsize=14)
                        plt.savefig("{}\\{}\\Pressure\\node{}_Pres_iter{}".format(mode, lambdaVal, emb_node_idx,it))
                        plt.close()
                    
                        if mode == 'gradient':
                            sns.heatmap(df_A, annot=False, vmin = 0, vmax = 0.025) # Fix y range
                        if mode == 'lrp.epsilon':
                            sns.heatmap(df_A, annot=False, vmin = -0.00035, vmax = 0.00035) # Fix y range
                        plt.title('{}(Acc), emb_node{}'.format(mode,emb_node_idx), fontsize=14)
                        plt.savefig("{}\\{}\\Acceleration\\node{}_Pres_iter{}".format(mode, lambdaVal, emb_node_idx,it))
                        plt.close()
                    
                        if mode == 'gradient':
                            sns.heatmap(df_R, annot=False, vmin = 0, vmax = 0.02) # Fix y range
                        if mode == 'lrp.epsilon':
                            sns.heatmap(df_R, annot=False, vmin = -0.00014, vmax = 0.00014) # Fix y range
                        plt.title('{}(Rot), emb_node{}'.format(mode,emb_node_idx), fontsize=14)
                        plt.savefig("{}\\{}\\Rotation\\node{}_Pres_iter{}".format(mode, lambdaVal, emb_node_idx,it))
                        plt.close()
                    
                        df_P.to_excel("{}\\{}\\Pressure\\node{}_Pres_iter{}.xlsx".format(mode,lambdaVal, emb_node_idx,it), index=False)
                        df_A.to_excel("{}\\{}\\Acceleration\\node{}_Acc_iter{}.xlsx".format(mode,lambdaVal, emb_node_idx,it), index=False)
                        df_R.to_excel("{}\\{}\\Rotation\\node{}_Rot_iter{}.xlsx".format(mode,lambdaVal, emb_node_idx,it), index=False)
                
                # Finally, average for all dimensions
                log_P = np.array(log_P)
                log_A = np.array(log_A)
                log_R = np.array(log_R)
                
                total_mean_P = np.mean(log_P, axis = 0)
                total_mean_A = np.mean(log_A, axis = 0)
                total_mean_R = np.mean(log_R, axis = 0)
                
                df_P_mean = pd.DataFrame(total_mean_P)
                df_A_mean = pd.DataFrame(total_mean_A)
                df_R_mean = pd.DataFrame(total_mean_R)
                
                if mode == 'gradient':
                    sns.heatmap(df_P_mean, annot=False, vmin = 0, vmax = 0.0055) # Fix y range
                if mode == 'lrp.epsilon':
                    sns.heatmap(df_P_mean, annot=False, vmin = -0.00015, vmax = 0.00015) # Fix y range
                plt.title('{}(Pres), tot_emb_node'.format(mode), fontsize=14)
                plt.savefig("{}\\{}\\Pressure\\tot_emb_node_Pres_iter{}".format(mode, lambdaVal,it))
                plt.close()
                df_P_mean.to_excel("{}\\{}\\Pressure\\tot_emb_node_Pres_iter{}.xlsx".format(mode,lambdaVal,it), index=False)
                
                if mode == 'gradient':
                    sns.heatmap(df_A_mean, annot=False, vmin = 0, vmax = 0.025) # Fix y range
                if mode == 'lrp.epsilon':
                    sns.heatmap(df_A_mean, annot=False, vmin = -0.00035, vmax = 0.00035) # Fix y range
                plt.title('{}(Acc), tot_emb_node'.format(mode), fontsize=14)
                plt.savefig("{}\\{}\\Acceleration\\tot_emb_node_Acc_iter{}".format(mode,lambdaVal,it))
                plt.close()
                df_A_mean.to_excel("{}\\{}\\Acceleration\\tot_emb_node_Acc_iter{}.xlsx".format(mode,lambdaVal,it), index=False)
                
                if mode == 'gradient':
                    sns.heatmap(df_R_mean, annot=False, vmin = 0, vmax = 0.02) # Fix y range
                if mode == 'lrp.epsilon':
                    sns.heatmap(df_R_mean, annot=False, vmin = -0.00014, vmax = 0.00014) # Fix y range
                plt.title('{}(Rot), tot_emb_node'.format(mode), fontsize=14)
                plt.savefig("{}\\{}\\Rotation\\tot_emb_node_Rot_iter{}".format(mode,lambdaVal,it))
                plt.close()
                df_R_mean.to_excel("{}\\{}\\Rotation\\tot_emb_node_Rot_iter{}.xlsx".format(mode,lambdaVal,it), index=False)    
                
            tf.reset_default_graph()
            session.close()
     
       
