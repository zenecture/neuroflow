# NeuroFlow

NeuroFlow is a lightweight library to construct, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with Breeze (+ NetLib for near-native performance).

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

# Introduction

This project consists of three modules:

- core: the neural network architecture
- application: plugins, helpers, functionality related to application
- playground: examples with resources
    
# Getting Started

For SBT-Usage, just add this GitHub repository to your dependencies. A maven (or similar) repository is planned for the future.
Also, you may have a look at the playground for some inspiration.

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=443 height=320 />

Let's construct the net depicted above. First, we have to pick the desired behavior:

```scala
import neuroflow.nets.DefaultNetwork._
import neuroflow.core.WeightProvider.randomWeights
import neuroflow.application.plugin.Style._
```

The first import gives us an implicit constructor for the default net implementation with gradient descent. 
The second one yields an implicit weight provider, which determines the initial weight values. The last one is eye candy to keep the notation a little shorter. 
The idea behind this 'import a-la-carte' is to change the underlying net implementation without changing the 'meat' of the code.

```scala
val fn = Sigmoid.apply
val net = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: Nil)
```

The whole architecture of the net is defined here. We want to use a sigmoid activation function `fn` for our hidden and output layers. 
Optionally, we could provide a `NetSettings` instance to force approximation of gradients or to disable verbosity. If we would need a more complex net, we would simply stack layers and functions:

```scala
val fn = Sigmoid.apply
val gn = Tanh.apply
val net = Network(Input(50) :: Hidden(20, fn) :: Hidden(10, gn) :: Output(2, fn) :: Nil)
```

Be aware that a default network must start with one `Input` layer and end with one `Output(i, fn)` layer. 

# Training

Let's train our net with the `train` method. It expects the inputs `xs` and their desired outputs `ys`. By design, the type signature is `Seq[Seq[_]]`, because this will promise the most general (Seq, List, Vector, ...) case in Scala.
Also, some rates and rules need to be defined, like precision or maximum iterations through a `TrainSettings` instance.

```scala
val xs = -->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = -->(->(0.0), ->(1.0), ->(1.0), ->(0.0))
val trainSets = TrainSettings(stepSize = 2.0, precision = 0.001, maxIterations = 10000)
net.train(xs, ys, trainSets)
```

During training, the derivatives of the net with respect to the weights are constructed, so the optimal weights can be computed. This can take a short (or very long) time, depending on the challenge. The learning progress will appear on console so we can track it. Bear in mind that a net is intended to be an atomic instance, so it is blocking and has mutable state inside concerning the training. An immutable net is infeasible, because it needs huge stack and heap sizes during training. In practical applications, multiple net instances form an overall net architecture, and this usually is the place for any parallelism.

# Evaluation

Our trained net can be evaluated with the `evaluate` method.

```scala
val result = net.evaluate(->(0.0, 0.0))
```

This will give us a result vector (kind `Seq[_]`) with the dimension of our specified output layer.

# IO

To keep the efforts of a hard, long training phase, it is important to save and load a net. The computed weights are the precious part of the net, so in the component `neuroflow.application.plugin.IO` we will find functionality to build a `WeightProvider` from IO. Scala Pickling is used as the (de-)serialization framework.

```scala
val file = "/path/to/net.json"
implicit val wp = File.read(file)
val net = Network(layers)
File.write(net, file)
```

Here, `File.read` will yield an implicit `WeightProvider` from file to construct a net.
Afterwards it will be saved to the same file with `File.write`. If the desired target is a database, simply use `Json` instead and save it on string-level.
However, to not dictate anything, all important types extend `Serializable`, so feel free to work with the bytes on your own.

# Next

- More network implementations like Shared Network, Constrained Network, ...
- Modularization of gradient techniques, as there are more solid ones than gradient descent