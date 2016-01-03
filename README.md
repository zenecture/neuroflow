# NeuroFlow

NeuroFlow is a lightweight library to construct, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with Breeze (+ NetLib for near-native performance).
!Use in production at own risk!

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=800 height=500 />

# Introduction

This project consists of three modules:

- core: the neural network architecture
- application: functionality and helpers related to certain challenges
- playground: examples with resources
    
# Getting Started

For SBT-Usage, just add this GitHub repository to your dependencies.
Also, you may have a look at the playground for some inspiration.

# Behind the scenes

A net is constructed with

```scala
val network = Network(Input(2) :: Hidden(3, Sigmoid.apply) :: Output(1, Sigmoid.apply) :: Nil)
network.train(Seq(Seq(0.0, 0.0), Seq(0.0, 1.0), Seq(1.0, 0.0), Seq(1.0, 1.0)), Seq(Seq(0.0), Seq(1.0), Seq(1.0), Seq(0.0)), 10.0, 0.001, 10000)
```

