# Explainable Gait Recognition with Prototyping Encoder Decoder
This repository is the official implementation of Explainable Gait Recognition with Prototyping Encoder–Decoder paper, which proposes a method to successfuly address the open set gait recognition problem.

The code here implements an encoder-decoder network architecture that learns mapping from the input (the gait information consistent of unit steps) to a latent space. The network utilizes two types of loss functions. The first one is based on the triplet loss function and it enforces that the distances in a latent space of homogeneous input pairs are smaller than those of heterogeneous input pairs. The second loss function minimizes the difference between reconstructed inputs and their corresponding prototypes. 

Also here is implemented the module that allows the analysis of which part of the input is relevant to the recognition performance by using explainable tools such as sensitivity analysis (SA) and layer-wise relevance propagation (LRP) which are available at the iNNvestigate toolbox (pypi.org/project/innvestigate/)

# Requirements
Some of the main packages used for this project are Tensorflow-gpu 1.14, Keras 2.2.4, innvestigate 1.08, and scikit-learn 0.23.2.
It is recommended to create a new environment and install the packages listed in requirements.txt:
```
pip install -r requirements.txt
```

# Datasets
As is shown in the following image, the data was collected from 30 subjects and it was split into three sets: 

<img src="images/split.png" width="70%" class="center">

- <b>Training set</b>: used to train the encoder-decoder network. It consists of all the unit steps of 16 individuals selected randomly.
- <b>Known-test set</b>: it contains the unit steps of 7 individuals selected randomly from the 14 remaining people after selecting the training set. This dataset is divided in two subsets. The first subset consists of 10 unit steps for each individual and it is used for training the OSVM classifier. The second subset is the remaining steps of the same 7 individuals and it is used to test the classifier as known data in the open set gait recognition problem.
- <b>Unknown test set</b>: it contains all the unit steps of the remaining 7 subjects which were not used in any training process, therefore they are unknown subjects. It is used for testing the classifier as unknown data in the open set gait recognition problem.

# Modules
To train the models in the paper, run this command:
```
cd code
python training.py
```
We repeat our experiment 10 times...

# Evaluation
The system is evaluated in terms of Accuracy (ACC), True Positive Rate (TPR), and True Negative Rate (TNR) defined as follows:

<table>
  <tr>
    <td>![equation one](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20ACC%20%3D%20%5Cfrac%7BTP%20&plus;%20TN%7D%7BTP%20&plus;%20FN%20&plus;%20TN%20&plus;%20FP%7D)</td>
    <td>![equation two](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20TPR%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D)</td>
    <td>![equation three](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20TNR%3D%20%5Cfrac%7BTN%7D%7BTN%20&plus;%20FP%7D)</td>
  </tr>
</table>

Where, 
- TP stands for True Positive and it is a unit step in the known test set that is classified correctly. 
- FN stands for False Negative and it is a unit step in the known test set that is classified incorrectly. 
- TN stands for True Negative and it is a unit step in the unknown test set that is classified correctly as an unknown participant.
- FP satnds for False Negative and it is a unit step in the unknown test set that is classified incorrectly as a known participant.

# Pre-trained Model
You can download pretrained model here: ...


# Results
Performance as function of lambda:
![Lambda](images/acc-lambda-v.png)



# Contributors
Nelson Minaya nelson.minaya@student.csulb.edu <br/>
Sella Bae sella.bae@student.csulb.edu <br/>

