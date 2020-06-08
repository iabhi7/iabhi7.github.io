---
title: "An insight into Neural network Activation functions"
layout: post
date: 2016-12-17 22:48
image: /assets/images/markdown.jpg
headerImage: false
tag:
  - Activation Function
  - Neural Network
  - Sigmoid
  - Tanh
  - Relu
  - Maxout
  - Leaky Relu
category: blog
author: jamesfoster
description: Are you a beginner in Neural Network confused about which activation function you should use, well here is a brief of the common activation functions that are used in Neural Network.
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

# A bit of Introduction



{:.text-center img}
![Structure of a simple neuron]({{ site.urlimg }}/simple-neuron.png)
<figcaption class="caption">Structure of a simple neuron</figcaption>


The basic building block of a neural network is a processing-unit which is linked to n input-units through a set of n directed connections. The single unit model is characterized by: 

- A threshold input value denoted by : x<sub>i</sub>. 
- A univariate activation function denoted in the figure
- A vector of "weights," denoted by w<sub>ij</sub>
- And ofcourse the output value.

But for better understanding lets discuss very briefly the biological system from which a large portion of this area has been inspired b

{:.text-center img}
![A image of a biological neuron]({{ site.urlimg }}/neuron.png "Toy example")


{:.text-center img}
![A mathematical model of a Neuron]({{ site.urlimg }}/neuron_model.png "Toy example")


The basic computational unit of the brain is a neuron. Approximately 86 billion neurons can be found in the human nervous system and they are connected with approximately 10^14 - 10^15 synapses. The diagram below shows a cartoon drawing of a biological neuron (left) and a common mathematical model (right). Each neuron receives input signals from its dendrites and produces output signals along its (single) axon. The axon eventually branches out and connects via synapses to dendrites of other neurons. In the computational model of a neuron, the signals that travel along the axons (e.g. x<sub>0</sub>) interact multiplicatively (e.g. w<sub>0</sub>x<sub>0</sub>) with the dendrites of the other neuron based on the synaptic strength at that synapse (e.g. w<sub>0</sub>). 

The idea is that the synaptic strengths (the weights w) are learnable and control the strength of influence (and its direction: excitory (positive weight) or inhibitory (negative weight)) of one neuron on another. In the basic model, the dendrites carry the signal to the cell body where they all get summed. If the final sum is above a certain threshold, the neuron can fire, sending a spike along its axon. In the computational model, we assume that the precise timings of the spikes do not matter, and that only the frequency of the firing communicates information. Based on this rate code interpretation, we model the firing rate of the neuron with an activation function f, which represents the frequency of the spikes along the axon.


So, in simple words a Activation fucntion of a node in terms of Neural Network defines the output of the node, given a set of predetermined inputs.


# History of Activation function.

The Mark 1 Perceptron machine was the first implementation of the perceptron algorithm, the first learning algorithm for neural network. This machine used a unit step function as a kind of "Activation Function"(No such term existed at that time.) But there was no concept of loss function and no concept of back propagation.
Then in late 60's a paper came out titled ["Perceptrons, Adalines and Backpropagation"](http://isl-www.stanford.edu/~widrow/papers/bc1995perceptronsadalines.pdf). This was the first multi layer perceptron network. Then in 1986 a research paper by Rumelhart came them emphasised the use of back propagation and after this the concept of back propagation became very popular, the paper also talked about loss function so this was a major advancement in Neural Network.
After this the domain of Neural Network remained quite for many many years. Though many small development took place in these year but it was late 2000's that boosted the world of Neural Network.
Here are links of some research paper that happened in the 2000's, (especially have a look at the Microsoft research paper of 2012)





# Commonly used Activation Function:

## **Sigmoid activation function**
A sigmoid function has a vast history and for a very long period of time Sigmoid was used and preferred over the rest it had a nice interpretation as the firing rate of a neuron. The sigmoid is a mathematical function having an "S" shaped curve (sigmoid curve). Often, sigmoid function refers to the special case of the logistic function shown in the first figure and defined by the formula

<div>
  $$\frac{n!}{k!(n-k)!}$$
</div>


\begin{align*}
  & {\displaystyle S(t)={\frac {1}{1+e^{-t}}}.}
\end{align*}
$$

The sigmoid non-linearity has the mathematical form $$ \begin{align*} & \sigma(x) = 1/(1+e^{-x}) \end{align*} $$ and is shown in the image above on the left. 


The sigmoid function takes a real-valued number and “squashes” it into range between 0 and 1. And this "Squasing" technique is one of the major drawbacks of the sigmoid function. In particular, large negative numbers become 0 and large positive numbers become 1. Let's see the major drawback of the Sigmoid activation function in detail:

- **Sigmoids saturate and kill gradients:** I just mentioned that the sigmoid function squashes a number in the range of 0 and 1 and as a result the neuron's activation saturates at wither 1 or 0. See the sigmoid curve once again, at large positive negative values the gradient is almost 0. Now during Backpropagation this gradients are supposed to be multiplied by the local gradient at each neuron and in this case the output of backpropagation will almost come out to be 0 (due to very small gradient value). Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data.

- **Sigmoid outputs are not zero-centered:** Many a times (almost everytime) it happens that the neuron receives data that is not zero-centered. In that case if the data coming into the neuron is always positive, the gradient on the weights 'w' will always be either positive or all negative during backpropagation(depending upon the gradient). Due to this the gradient updates will be very random (zig-zag shape to be precise) which is very undesirable. 

- **Slow convergence:** The sigmoid activation function has a very slow convergence. So the initialization of the weights become very important if we are using the sigmoid activation function. (The reason can be inferred from the above two points), if the initial weights are too large then most of the neurons would saturate and the network will barely learn. (Tip: Start with a very small 'w')

- **Expensive:** The sigmoid function uses the exponential function, which is computationally expensive as compared to other activation functions.

Here is a toy example of an implementation of Sigmoid

{% highlight python %}
import numpy as np

# sigmoid function with forward and back prop
def nonlin(x,deriv=False):
  if(deriv==True):
    return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
# feel free to play with the imput dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

<!-- # initialize weights randomly with mean 0 -->
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1



{% endhighlight %}



**Verdict:** Don't ever use Sigmoid, never ever, never ever ever....


## **Tanh Activation function**

The tanh non-linearity is shown on the image below. Tanh is nothing but $$ \begin{align*} & \textrm{tanh}(x) = 2\sigma(2x) - 1 \end{align*} $$. Tanh came into the picture with the goal of answering one of the problems of Sigmoid activation function(Point no 2 of Sigmoid). It is zero-centered and squashes a real-valued number to the range [-1, 1]. Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron its output is zero-centered. Although tanh activation function is nothing but 2*sigmoid - 1, but for a better understanding of the advantages of tanh you can have a look at [this paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) 

{:.text-center img}
![A tanh graph, see the range [-1, 1]]({{ site.urlimg }}/tanh.png "Toy example")

**Verdict:** Always prefer tanh non-linearity over the sigmoid nonlinearity. But you should avoid using Tanh too. (as there are more problems to be addressed to)


## **ReLU Activation function**

The Rectified Linear Unit is one of the most popular and widely used activation function. The ReLU is a simple thresholding function f(x)=max(0,x). Let's see the pros and cons of the ReLU activation function.

- **Faster Convergence rate:** The convergence of the gradient is accelerated by a factor of 6 as compared to the sigmoid/tanh functions. It is argued that this is due to its linear, non-saturating form. 

- **Computationally cheap:** The ReLU activation function simply thresholds the input so it is computationally very cheap.
Gradient kill or Dying ReLU: As you can see that the ReLU is 0 for all negative values so it is possible that a large gradient flowing through the unit can never get activated and remain 'dead' for the whole time, this is also referred to as the 'dying ReLU' problem. So the ReLU units can be fragile during training and can “die”. Though we can tackle this problem to some extent with proper setting of the Learning rate. [Understanding Deep Neural Networks with Rectied Linear Units](https://arxiv.org/pdf/1611.01491v3.pdf) gives a better insight into the Relu networks.

- The RelU activation function is also non-zero centered.

{:.text-center img}
![A Relu graph]({{ site.urlimg }}/relu.png "Toy example")

**Verdict:** If you are a beginner in Neural Network then the ReLU activation function should be your default choice.


## **Leaky ReLU Activation function**

ReLu activation function had this major "dying ReLU" problem and the leaky ReLUs are one attempt to fix the “dying ReLU” problem. Here for all negative values of x (x < 0), the leaky ReLU have a very small negative slope. That is, the function computes f(x)=𝟙(x<0)(αx)+𝟙(x>=0)(x)
where α is a small constant. In some problems this activation function works wonder but it is not always very consistent. You are more than encouraged to try this on your own.
Interesting Point: The constant α can also be made as one of the parameters of each neuron and this can be learned by the network. Such editing are also termed as PReLU neurons, you can see a lot more about this in [Delving Deep into Rectifiers, by Kaiming He et al., 2015](https://arxiv.org/pdf/1502.01852.pdf). (Warning: It has 3 pages of mathematical derivation)


## **Maxout**

Other types of units have been proposed that do not have the functional form f(wTx+b)
where a non-linearity is applied on the dot product between the weights and the data. One relatively popular choice is the Maxout neuron (introduced recently by Goodfellow et al.) that generalizes the ReLU and its leaky version. The Maxout neuron computes the function max(wT1x+b1,wT2x+b2). Notice that both ReLU and Leaky ReLU are a special case of this form (for example, for ReLU we have w1,b1=0). The Maxout neuron therefore enjoys all the benefits of a ReLU unit (linear regime of operation, no saturation) and does not have its drawbacks (dying ReLU). However, unlike the ReLU neurons it doubles the number of parameters for every single neuron, leading to a high total number of parameters. Also you need to have a bit of knowledge about Dropouts for implementing Maxout. Here is a [paper](https://arxiv.org/pdf/1302.4389v4.pdf) on maxout activation function.



# Conclusion:

There is always this question/doubt that every beginner Neural Network enthusiast has is "What activation function should I use?". I would say there is always this perfect amalgamation between the type of network and the activation function and its a mix and match stuff butthe default choice should always be ReLU non-linearity but we have to be very carefull about the dying unit problem. Also you are welcomed to experiment on the Leaky ReLU or the Maxout activation function, these activation function may give you better reults but they have not established themselves in the industry. You can also try out Tanh but the reults would be worse. Sigmoid? Seriously, are you even considering using Sigmoid? If yes, Go ahead and test it :P 







### References


Refer to the following docs/videos for more information.

1: [CS231n: Andrej Kapathy explanation is the best](http://cs231n.github.io). (I took a lot from cs231n page for writing this blog.)

Want to see something else added? <a href="https://github.com/dyndna/lanyon-plus/issues/new">Open an issue.</a> Feel free to contribute.


P.S Jupyter notebook for each activation function is coming soon.. Stay tuned.
