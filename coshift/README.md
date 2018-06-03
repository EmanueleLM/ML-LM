## Infonet
# Evaluating the mutual information flow in a neural network

## Things implemented so far, 04/02/2018: ##

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);
  - all from backpropagation to check gradient routine is done by using just numpy
  - add with a human-like line the topology specification (otherwise it's considered fully connected)

- calculate the mutual information between the discrete distribution of the input and the output.


TODO (in order of importance to me):

- fasten the mutual information calculation between adjacent and non-adjacent layers.

## Create a network ##
Let's suppose we want to build a 3 layers (input + 2 hidden layers) neural net, with sigmoid in the first hidden layer,
exp in the second layer and a leakyrelu in the last layer (let's suppose its a good architecture :-S ). We want an input which is 
10-dimensional, 35 neurons in the first hidden layer, 40 neurons in the second hidden layer and 5 neurons in the output layer.
We will use the L2 error measure.
We will have:
```python
import deepnet as dn
net = dn.DeepNet(10, np.array([[35, "sigmoid"], [40, "exp"], [5, "leakyrelu"]]), "L2");
```

We want to specify the topology of the net, in such a way that the input is connected in this way:
- the inputs from the first to the fifth are connected to the first 20 neurons of the first hidden layer
- the 6th and 7th inputs are connected, respectively, to the 21th and 22th neurons of the first hidden layer; 
- the rest of the inputs are fully connected to the rest of the neurons of the first hidden layer;
- the rest of the net is fully connected (just to have few text to read in this tutorial, the way you connect the other layers is very the same as what I described above).
```python
net.net_topology('layer(1): :5|:20, 6|21, 7|22, 8:|23: layer(2): :|: layer(3): :|:'); 
```

The code exploit a well known propery of mutual information: if the two distributions are s.t. Y=f(X), where f is a function, it holds that I(X,Y)=H(f(x)).
So, if you want to calculate the mutual information between input and output, you first need to discretize the output function (which is from the theory point of view, continuous) in discrete bins, then you can calculate the mutual information easily. Here's the code you need to use:
```python
import infolayer as info
il = info.InfoLayer(sample_size, net.partial_activations[1], method_output="uniform", bins_output=5);
mutual_information = il.entropy();
```
