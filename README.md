# Explainable Gait Recognition with Prototyping Encoder Decoder
This repository is the official implementation of Explainable Gait Recognition with Prototyping Encoder–Decoder paper, which proposes a method to successfuly address the open set gait recognition problem.

According to the paper, the code here implements a encoder-decoder network architecture that learns mapping from the input (the gait information consistent of unit steps) to a latent space. The network utilizes two types of loss functions. The first one is based on the triplet loss function and it enforces that the distances in a latent space of homogeneous input pairs are smaller than those of heterogeneous input pairs. The second loss function minimizes the difference between reconstructed inputs and their corresponding prototypes. 

Also here is implemented the module that allows the analysis of which part of the input is relevant to the recognition performance by using explainable tools such as sensitivity analysis (SA) and layer-wise relevance propagation (LRP) which are available at the iNNvestigate toolbox (https://pypi.org/project/innvestigate/)

# Requirements
The main packages used for this project are Tensorflow-gpu version 1.14, Keras 2.24, innvestigate 1.08, and scikit-learn 0.23.2.
It is recommended to create a new environment and install the packages listed in requirements.txt:
```
pip install -r requirements.txt
```
# Training
To train the models in the paper, run this command:
```
cd code
python training.py
```
We repeat our experiment 10 times...

# Evaluation
We divide our test set into ...



# Pre-trained Model
You can download pretrained model here: ...


# Results
..
# Contributors
Nelson Minaya nelson.minaya@student.csulb.edu <br/>
Sella Bae sella.bae@student.csulb.edu <br/>

