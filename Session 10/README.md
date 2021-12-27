## Team

Yuvaraj V (yuvaraj100493@gmail.com)

Satwik Mishra (satwiksmishra@gmail.com)


## Assignment Part A

### Goals

1.Train ResNet-18 on Tiny Image dataset.
2.Apply augmenttions using Albumentations.
3.Target >50% testing/validation accuracy in 50 Epochs.

### Necessary Packages and Modules

All the necessary modules/functions can be found at: https://github.com/satwikmishra/pytorch_modules/tree/main/src

### Analysis

I observed that there is significant overfitting when I trained it for 30 epochs. I had initially trained it for 40 epochs but the accuracy numbers did not improve. 
We can see in the logs that training accuracy is 97.86% and testing accuracy is only 39.20%

### Plots

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/plots.PNG)

```python

```

### logs
EPOCH: 0
  0%|          | 0/196 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=1.3992035388946533 Batch_id=195 Accuracy=38.43: 100%|██████████| 196/196 [08:05<00:00,  2.48s/it]
Test set: Average loss: 2.3305, Accuracy: 2462/10000 (24.62%)

EPOCH: 1
Loss=1.101202130317688 Batch_id=195 Accuracy=57.34: 100%|██████████| 196/196 [08:13<00:00,  2.52s/it]
Test set: Average loss: 3.3094, Accuracy: 1899/10000 (18.99%)

EPOCH: 2
Loss=1.0850259065628052 Batch_id=195 Accuracy=67.13: 100%|██████████| 196/196 [08:07<00:00,  2.49s/it]
Test set: Average loss: 2.2819, Accuracy: 3422/10000 (34.22%)

EPOCH: 3
Loss=0.6462981104850769 Batch_id=195 Accuracy=73.73: 100%|██████████| 196/196 [08:14<00:00,  2.52s/it]
Test set: Average loss: 3.4135, Accuracy: 2460/10000 (24.60%)

EPOCH: 4
Loss=0.6057966351509094 Batch_id=195 Accuracy=77.38: 100%|██████████| 196/196 [08:07<00:00,  2.48s/it]
Test set: Average loss: 2.4384, Accuracy: 3558/10000 (35.58%)

EPOCH: 5
Loss=0.6364989280700684 Batch_id=195 Accuracy=80.68: 100%|██████████| 196/196 [08:14<00:00,  2.52s/it]
Test set: Average loss: 3.3690, Accuracy: 2979/10000 (29.79%)

EPOCH: 6
Loss=0.591964066028595 Batch_id=195 Accuracy=82.89: 100%|██████████| 196/196 [08:05<00:00,  2.48s/it]
Test set: Average loss: 3.1733, Accuracy: 2651/10000 (26.51%)

EPOCH: 7
Loss=0.41414475440979004 Batch_id=195 Accuracy=84.88: 100%|██████████| 196/196 [08:12<00:00,  2.51s/it]
Test set: Average loss: 3.2165, Accuracy: 2679/10000 (26.79%)

EPOCH: 8
Loss=0.22476211190223694 Batch_id=195 Accuracy=86.28: 100%|██████████| 196/196 [08:05<00:00,  2.48s/it]
Test set: Average loss: 3.1108, Accuracy: 3022/10000 (30.22%)

EPOCH: 9
Loss=0.3741905093193054 Batch_id=195 Accuracy=87.26: 100%|██████████| 196/196 [08:12<00:00,  2.51s/it]
Test set: Average loss: 3.4546, Accuracy: 2637/10000 (26.37%)

EPOCH: 10
Loss=0.30848950147628784 Batch_id=195 Accuracy=88.55: 100%|██████████| 196/196 [08:05<00:00,  2.48s/it]
Test set: Average loss: 3.6618, Accuracy: 2737/10000 (27.37%)

EPOCH: 11
Loss=0.33385854959487915 Batch_id=195 Accuracy=89.32: 100%|██████████| 196/196 [08:12<00:00,  2.51s/it]
Test set: Average loss: 2.5134, Accuracy: 3846/10000 (38.46%)

EPOCH: 12
Loss=0.4112222194671631 Batch_id=195 Accuracy=90.00: 100%|██████████| 196/196 [08:07<00:00,  2.48s/it]
Test set: Average loss: 2.6552, Accuracy: 3831/10000 (38.31%)

EPOCH: 13
Loss=0.2877618968486786 Batch_id=195 Accuracy=91.04: 100%|██████████| 196/196 [08:11<00:00,  2.51s/it]
Test set: Average loss: 3.1734, Accuracy: 3311/10000 (33.11%)

EPOCH: 14
Loss=0.0998789519071579 Batch_id=195 Accuracy=94.40: 100%|██████████| 196/196 [08:04<00:00,  2.47s/it]
Test set: Average loss: 2.8302, Accuracy: 3782/10000 (37.82%)

EPOCH: 15
Loss=0.17392653226852417 Batch_id=195 Accuracy=95.38: 100%|██████████| 196/196 [08:11<00:00,  2.51s/it]
Test set: Average loss: 2.8188, Accuracy: 3829/10000 (38.29%)

EPOCH: 16
Loss=0.19943904876708984 Batch_id=195 Accuracy=95.82: 100%|██████████| 196/196 [08:04<00:00,  2.47s/it]
Test set: Average loss: 2.8196, Accuracy: 3835/10000 (38.35%)

EPOCH: 17
Loss=0.1265377700328827 Batch_id=195 Accuracy=95.98: 100%|██████████| 196/196 [08:15<00:00,  2.53s/it]
Test set: Average loss: 2.9378, Accuracy: 3760/10000 (37.60%)

EPOCH: 18
Loss=0.11288662254810333 Batch_id=195 Accuracy=96.22: 100%|██████████| 196/196 [08:06<00:00,  2.48s/it]
Test set: Average loss: 3.0157, Accuracy: 3769/10000 (37.69%)

EPOCH: 19
Loss=0.05660974234342575 Batch_id=195 Accuracy=96.40: 100%|██████████| 196/196 [08:12<00:00,  2.51s/it]
Test set: Average loss: 3.1033, Accuracy: 3651/10000 (36.51%)

EPOCH: 20
Loss=0.11492566019296646 Batch_id=195 Accuracy=96.57: 100%|██████████| 196/196 [08:06<00:00,  2.48s/it]
Test set: Average loss: 2.9370, Accuracy: 3805/10000 (38.05%)

EPOCH: 21
Loss=0.09111280739307404 Batch_id=195 Accuracy=96.75: 100%|██████████| 196/196 [08:13<00:00,  2.52s/it]
Test set: Average loss: 3.0445, Accuracy: 3740/10000 (37.40%)

EPOCH: 22
Loss=0.12601251900196075 Batch_id=195 Accuracy=96.92: 100%|██████████| 196/196 [08:06<00:00,  2.48s/it]
Test set: Average loss: 3.0257, Accuracy: 3852/10000 (38.52%)

EPOCH: 23
Loss=0.15573230385780334 Batch_id=195 Accuracy=97.03: 100%|██████████| 196/196 [08:14<00:00,  2.52s/it]
Test set: Average loss: 3.0294, Accuracy: 3771/10000 (37.71%)

EPOCH: 24
Loss=0.08490946143865585 Batch_id=195 Accuracy=97.17: 100%|██████████| 196/196 [08:07<00:00,  2.49s/it]
Test set: Average loss: 3.0268, Accuracy: 3859/10000 (38.59%)

EPOCH: 25
Loss=0.05165805667638779 Batch_id=195 Accuracy=97.52: 100%|██████████| 196/196 [08:14<00:00,  2.52s/it]
Test set: Average loss: 3.0788, Accuracy: 3820/10000 (38.20%)

EPOCH: 26
Loss=0.05589020997285843 Batch_id=195 Accuracy=97.67: 100%|██████████| 196/196 [08:06<00:00,  2.48s/it]
Test set: Average loss: 3.0585, Accuracy: 3875/10000 (38.75%)

EPOCH: 27
Loss=0.14563396573066711 Batch_id=195 Accuracy=97.65: 100%|██████████| 196/196 [08:12<00:00,  2.51s/it]
Test set: Average loss: 3.0499, Accuracy: 3883/10000 (38.83%)

EPOCH: 28
Loss=0.1375075727701187 Batch_id=195 Accuracy=97.86: 100%|██████████| 196/196 [08:05<00:00,  2.48s/it]
Test set: Average loss: 3.0242, Accuracy: 3920/10000 (39.20%)

EPOCH: 29
Loss=0.07928366959095001 Batch_id=195 Accuracy=97.81: 100%|██████████| 196/196 [08:11<00:00,  2.51s/it]
Test set: Average loss: 3.1237, Accuracy: 3847/10000 (38.47%)


```python

```

## Assignment Part B

COCO is a large-scale object detection, segmentation, and captioning dataset.

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/dataset_examples.PNG)


### Class distribution

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/elbow_method.PNG)

```python

```

### K means clustering and anchor boxes
![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/cluster_bounding_box.PNG)

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/anchor_3.PNG)

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/anchor_4.PNG)

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/anchor_5.PNG)

![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/anchor_6.PNG)


```python

```

### K-means elbow method
![alt text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/elbow_method.PNG)

```python

```

