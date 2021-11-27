# Problem Statement


Train ResNet18 for 40 Epochs

20 misclassified images

20 GradCam output on the SAME misclassified images

Apply these transforms while training:

a. RandomCrop(32, padding=4)

b. CutOut(16x16)

c. Rotate(±5°)

Must use ReduceLROnPlateau

Must use LayerNormalization ONLY

# Solution and Approach

## Link to repository: https://github.com/satwikmishra/pytorch_modules. This repository contains all the necessary modules required for the assignments from here on. 
