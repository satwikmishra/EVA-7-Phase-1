Transformers, which are based on self-attention, have become the default model for natural language processing (NLP). In general, large texts are pre-trained on, and then fine-tuned on, a smaller task-specific dataset (Devalin et al., 2019). This is due to Transformers' speed and scalability. Convolutional architectures, however, remain popular in computer vision. Because Natural Language Processing has been so successful, researchers have been using CNN-like architectures and self-attention simultaneously. 

## Vision transformers with PyTorch

It is a goal of the project to train Vision Transformers to classify dogs and cats.

## Dataset

The data was downloaded from Kaggle.

The train folder contains 25000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog or cat (1 = dog, 0 = cat)

![alt_text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/capture_dog.PNG)


## Model

https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/VIT/VIT.ipynb
```python

```

## Training log
Epoch : 42 - loss : 0.5541 - acc: 0.7127 - val_loss : 0.5412 - val_acc: 0.7263

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 43 - loss : 0.5583 - acc: 0.7110 - val_loss : 0.5500 - val_acc: 0.7164

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 44 - loss : 0.5574 - acc: 0.7128 - val_loss : 0.5464 - val_acc: 0.7217

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 45 - loss : 0.5525 - acc: 0.7119 - val_loss : 0.5494 - val_acc: 0.7120

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 46 - loss : 0.5517 - acc: 0.7136 - val_loss : 0.5469 - val_acc: 0.7097

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 47 - loss : 0.5510 - acc: 0.7165 - val_loss : 0.5485 - val_acc: 0.7225

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 48 - loss : 0.5462 - acc: 0.7196 - val_loss : 0.5431 - val_acc: 0.7201

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 49 - loss : 0.5519 - acc: 0.7161 - val_loss : 0.5403 - val_acc: 0.7231

  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 50 - loss : 0.5508 - acc: 0.7142 - val_loss : 0.5352 - val_acc: 0.7235

```python

```
