```python

```

## Assignment objective


```python
99.4% validation accuracy
Less than 20k Parameters
You can use anything from above you want. 
Less than 20 Epochs
```

## Some research work

Dropout is a method where randomly selected neurons are dropped during training. 
Link: http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf
        
Batch Normalization: It normalizes (changes) all the input before sending it to the next layer. This is very similar to the normalization technique we use. 

Average Pooling is a pooling operation that calculates the average value for patches of a feature map which is different from max pooling which calculates the maximum value for each patch of the feature map.

What is the order of placing the Dropout, Batch-normalization and activation function? 

CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC or Pooling
Source: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

## Architecture of the model

The model includes Batch Normalization, Drop-out, Global Average Pooling. The total number of parameters are 17200.
The order in which they are supposed to be placed is mentioned above.

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/snippet_no_of_parameters.PNG)

```python

```

## Logs

We have achieved 99.29% accuracy in this architecture using 17,200 parameters. 

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/training_log.PNG)

## Resources 

https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99

https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf

https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721


