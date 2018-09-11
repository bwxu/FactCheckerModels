# FactCheckerModels
## Introduction
This code contains several neural models using Bi-LSTM and CNN architectures 
applied on the LIAR dataset. The data from the LIAR dataset is derived from 
Politifact and are categorized into 6 labels: 

"pants-fire"  
"false"  
"barely-true"  
"half-true"  
"mostly-true"  
"true"  

The original paper describing the dataset can be found here:
https://arxiv.org/abs/1705.00648

## How to Run Models
First, download the train, eval, and test sets from the following link and 
save them in the data folder 
  https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
Also, download your favorite embeddings and add them to the data folder also.
  Word2Vec on Google News: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
  GloVe: https://nlp.stanford.edu/projects/glove/
Following the downloads, be sure to set the Location of data files section of
var.py appropriately.

Then, go to var.py and specify what model and what parameters you want to use
for the model. Ensure that the "FOLDER_NAME" is set properly to where you want
to save the trained model checkpoints.

After training the model, run test_model.py in order to evaluate it's
performance on the eval and test sets. Note that var.py when running
test_model.py should be the same as the var.py originally used during training.
A copy of the var.py used during training is saved to the specified 
"FOLDER_NAME" during training for reference.

## Requirements
This pip freeze list should represent a superset of the requirements needed to
run this code.

absl-py==0.1.11  
astor==0.6.2  
bleach==1.5.0  
boto==2.48.0  
boto3==1.6.6  
botocore==1.9.6  
bz2file==0.98  
certifi==2018.1.18  
chardet==3.0.4  
docutils==0.14  
gast==0.2.0  
gensim==3.4.0  
grpcio==1.10.0  
h5py==2.8.0  
html5lib==0.9999999  
idna==2.6  
jmespath==0.9.3  
json-lines==0.3.1  
Keras==2.2.2  
Keras-Applications==1.0.4  
Keras-Preprocessing==1.0.2  
Mako==1.0.7  
Markdown==2.6.11  
MarkupSafe==1.0  
nltk==3.2.5  
numpy==1.14.1  
protobuf==3.5.2  
python-dateutil==2.6.1  
PyYAML==3.12  
requests==2.18.4  
s3transfer==0.1.13  
scikit-learn==0.19.1  
scipy==1.0.0  
six==1.11.0  
sklearn==0.0  
smart-open==1.5.6  
tensorboard==1.6.0  
tensorflow-gpu==1.5.0  
tensorflow-tensorboard==1.5.1  
termcolor==1.1.0  
Theano==1.0.1  
urllib3==1.22  
Werkzeug==0.14.1  

## File & Folder Descriptions
**data**: folder used to store data such as embeddings data and train, eval, test datasets  
**definitions_of_models.py**: contains keras definitions of all neural models used  
**misc**: folder containing code for a simple "max credit" model that predicts labels
      corresponding to the most frequent past labels from the same speaker  
**models**: folder used as location to save trained models  
**parse_data.py**: contains functions used for data processing  
**test_model.py**: code used to test trained models loading them and predicting the eval 
               and test sets.  
**train_model.py**: code used to train models in accordance with parameters in var.py  
**var.py**: file used to indicate what parameters should be used to train or test  

