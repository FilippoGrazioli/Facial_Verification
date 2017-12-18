# Quick guide

## 1. Download pretrained Inception-ResNet 
https://drive.google.com/open?id=1ucEvrbOrqUJO_v80aifVXzg2xb3oA2-J

## 2. Place the pre-trained model in the model_checkpoints folder 

## 3. Embeddings of a set of known identities (e.g. You)
a. Take more or less 4 pictures of the people you want to recognize 
b. Place them in the known_identities folder like this:

-known_identites
  --Identity_1
  --Identity_2
  
c. Run '''export_embeddings.py''' and copy paste the calculated embeddings in the respective identity folder

## Run 
'''main.py'''

