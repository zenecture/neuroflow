# NeuroFlow

NeuroFlow is a lightweight library to construct, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with Breeze (+ NetLib for near-native performance).
!Use in production at own risk!

<p style="text-align: center;">
    <img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=500 height=500 />
</p>

# Introduction

This project consists of three modules:

- core: the neural network architecture
- application: functionality and helpers related to certain challenges
- playground: examples with resources
    
# Getting Started

For SBT-Usage, just add this GitHub repository to your dependencies.
Also, you may have a look at the playground for some inspiration.

# Building a Net

A net is constructed using a list of layers

```scala
val fn = Sigmoid.apply
val network = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: Nil)
```
The whole architecture of the net is defined here. For instance, 
if you want to use more hidden layers, just concatenate them.

```scala
Hidden(25, fn) :: Hidden(12, fn) :: Hidden(3, fn) :: Nil
```

Be aware that a network must start with one`Input` layer and end with one `Output(i, fn)` layer.

# Training

You can train a `Network` with the `train` method. It expects the inputs `xs` and their desired outputs `ys`.
Also, some rates and rules need to be defined, like precision or maximum iterations.

```scala
val xs = Seq(Seq(0.0, 0.0), Seq(0.0, 1.0), Seq(1.0, 0.0), Seq(1.0, 1.0))
val ys = Seq(Seq(0.0), Seq(1.0), Seq(1.0), Seq(0.0))
val learningRate = 0.01
val precision = 0.001
val maxIterations = 1000
network.train(xs, ys, learningRate, precision, maxIterations)
```

During training, the derivatives of the net with respect to the weights are constructed, 
so the optimal weights can be computed iteratively (gradient descent). The learning progress will appear on console so you can see what is going on.

# Evaluation

A trained `Network` can be evaluated with the `evaluate` method.

```scala
network.evaluate(Seq(0.0, 0.0))
```

# IO

Saving and loading a trained net is TODO. 