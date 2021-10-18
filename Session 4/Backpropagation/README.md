# Backpropagation

## How do machines really learn? 

Consider a group of bots built by a worker-bot in the factory is assigned to a teacher-bot. These tiny bots go to school and are tasked with recognizing a cat and a dog from a set of images. Now, the set of images and the answer keys are handed over to the teacher-bot who is as naive as the student-bots. The teacher-bot, having the answer keys, starts testing the student-bots after a small "teaching" session. They're very bad at what they do, all of them. However, with each iteration, the best of the student-bots are retained (those who scored better than their peers) and the rest are expelled. Yes,they're expelled from the school and their fate is unknown. Again, the student-bots go through the gruelling test assigned by the teacher-bots ( who has the answer keys ). This cycle of teaching, test, expel, repeat is continued over and over again until the student-bots have learnt to say a cat from a dog with a certain confidence. Remember, they did not know anything at first and with enough iterations, learning and "expelling" they were able to do the job of recognizing a cat from a dog. This is a very high-level overview of how the machine learns.

This is somewhat an analogy of backpropagation too. At first, when you compare your answers with the solutions, you might get a big fat zero because you'd have guessed all of them. But gradually, over time, say after 1000's of similar tests, you figure what to tweak so as to get a better score. In this way, epoch after epoch, with you getting smarter, your answers improve over time. At some point you learn to get the answers right! 

Now let us continue with the assignment! 

The goal of backpropagation is to optimise the weights so that the accuracy of the model improves!

# Assignment

## Forward propagation

A neural network with two inputs, two hidden neurons, two output neurons and bias = False. 


```python
h1, h2 --> Hidden layer neurons. 
w1,w2,w3,w4 --> weights
σ --> Sigmoid function
E1,E2 --> Errors
```

#### To calculate h1 and h2 we will need to pass the inputs through the network. Below is the technique to find h1 and h2. 


```python
h1 = w1*i1+w2*i2
h2 = w3*i1+w4*i2

```

#### The output that we obtain from h1 and h2 are passed to a sigmoid function to add non-linearity without which the model doesn't learn a thing. 


```python
a_h1 = σ(h1) = 1/(1+exp(-h1))
a_h2 = σ(h2) = 1/(1+exp(-h2))

```

#### We repeat the same process for calculating the output layer neurons. We use the output from the hidden layer activated neurons as input. Then the output of this is passed to a sigmoid function and we get a_o1 and a_o2


```python
o1 = w5*a_h1+w6*a_h2
o2 = w7*a_h1+w8*a_h2
a_o1 = σ(o1) = 1/(1+exp(-o1))
a_o2 = σ(o2) = 1/(1+exp(-o2))

```

#### Calculating the loss

The errors (E1,E2) are calculated from each output neurons (a_o1, a_o2) and E_total is the sum of the two errors. 


```python
E1 = 0.5*(t1-a_o1)²
E2 = 0.5*(t2-a_o2)²
E_Total = E1+E2

```

When we differentiate the error term we get 2*... which we need to cancel. So we multiply the result with 1/2

## Backward propagation

### Refer to the analogy at the top to understand the intuition. 

#### We calculate the partial derivative of E_total w.r.t w5,w6,w7 and w8. 


```python
∂E_total/∂w5 =∂(E1+E2)/∂w5=∂E1/∂w5=(∂E1/∂a_o1)*(∂a_o1/∂o1)*(∂o1/∂w5)
∂E1/∂a_o1=∂(0.5*(t1-a_o1)^2)/∂a_o1 = (t1-a_o1)*(-1)=a_o1-t1
∂a_o1/∂o1=∂(σ(o1))/∂o1=σ(o1)*(1-σ(o1))=a_o1*(1-a_o1)
∂o1/∂w5=a_h1

∂E_total/∂w5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
∂E_total/∂w6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
∂E_total/∂w7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
∂E_total/∂w8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2

```

#### Then we calculate the partial derivative of E_total w.r.t w1,w2,w3 and w4. 


```python
∂E1/∂a_h1=∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂a_h1=(a_o1-t1)*a_o1*(1-a_o1)*w5
∂E2/∂a_h1=(a_o2-t2)*a_o2*(1-a_o2)*w7
∂E_total/∂a_h1=∂E1/∂a_h1+∂E2/∂a_h1=((a_o1-t1)*a_o1*(1-a_o1)*w5) +((a_o2-t2)*a_o2*(1-a_o2)*w7)
∂E_total/∂a_h2=∂E1/∂a_h2+∂E2/∂a_h2=((a_o1-t1)*a_o1*(1-a_o1)*w6) +((a_o2-t2)*a_o2*(1-a_o2)*w8)
∂E_total/∂w1=(∂E_total/∂a_h1)*(∂a_h1/∂h1)*(∂h1/∂w1)
∂E_total/∂w1=∂E_total/∂a_h1*∂a_h1/∂h1*∂h1/∂w1=∂E_total/∂a_h1*a_h1*(1-a_h1)*∂h1/∂w1

∂E_total/∂w1=∂E_total/∂a_h1*a_h1*(1-a_h1)*i1
∂E_total/∂w2=∂E_total/∂a_h1*a_h1*(1-a_h1)*i2
∂E_total/∂w3=∂E_total/∂a_h2*a_h2*(1-a_h2)*i1
∂E_total/∂w4=∂E_total/∂a_h2*a_h2*(1-a_h2)*i2
```

#### After getting gradients for the weights w.r.t the total error: E_total, we subtract this value from the current weight by multiplication with learning rate to achieve updated weights.


```python
w1 = w1-learning_rate * ∂E_total/∂w1
w2 = w2-learning_rate * ∂E_total/∂w2
w3 = w3-learning_rate * ∂E_total/∂w3
w4 = w4-learning_rate * ∂E_total/∂w4
w5 = w5-learning_rate * ∂E_total/∂w5
w8 = w6-learning_rate * ∂E_total/∂w6
w7 = w7-learning_rate * ∂E_total/∂w7
w8 = w8-learning_rate * ∂E_total/∂w8
```

## Graph from the excel sheet after changing the values of Learning Rate. 

