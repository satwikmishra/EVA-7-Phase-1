# Problem statement


```python
1. Take a look at this post, which explains how to fine-tune DETR on a custom dataset.
2. Replicate the process and train the model yourself. Everything is mentioned in the post. The objectives are:
i.to understand how fine-tuning works
ii.to understand architectural related concepts
iii.be ready to format "your" dataset into something that can be used for custom fine-tuning of DETR.
iv.Expecting a readme file (along with Colab link in this README file) that:
    a.explains DETR
    b.explains encoder-decoder architecture
    c.explains bipartite loss, and why we need it
    d.explains object queries
    e.shows your results (your model on your dataset)
v.Expecting a notebook imported from Google Colab
vi.Please note that the deadline is exactly 1 week, and cannot be extended as your Capstone Deadline will start from next week.
```

## DETR

The DEtection TRansformer (DETR) is an object detection model, developed by Facebook Research. It utilizes Transformers with a backbone of ResNet50. DETR predicts all all objects at once and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects. Unlike most existing detection methods, DETR doesn't require any customized layers and can be reproduced easily in any framework that contains standard CNN and transformer classes.
![alt_text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/detr_transformer.png)


Here is a summary of how DETR works:

1. The image is passed through a backbone network (ResNet50) which gives a set of image features. The pretrained CNN backbone, which produces a set of lower dimensional set of features. These features are then scaled and added to a positional encoding, which is fed into a Transformer consisting of an Encoder and a Decoder.
2. Having width W and height H (In practice, we set C=2048, W=W₀/32 and H=H₀/32).
3. The image features pass through a Transformer which consists of an encoder and a decoder.
4. The output of decoder is 100 values which when passed through a FFN ( Feed Forward Network), give the prediction bbox.
5. While training, the output is passed through a bipartite matching loss function between the predicted bounding boxes and ground-truth boxes. This is because the predictions made are out of order.

## Encoder-Decoder Architecture
![alt_text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/architecture.png)

## Bipartite loss

Bipartite matches are one-to-one matches. For DETR, the output is not ordered, so a one-by-one matching of the predicted and ground truth labels is required to find the loss.

## Results

![alt_text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/res_2.png)

![alt_text](https://github.com/satwikmishra/EVA-7-Phase-1/blob/main/Images/res_3.png)

```python

```
