# NeuroFlow

NeuroFlow is a lightweight library to construct, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with Breeze (+ NetLib for near-native performance).

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=707 height=190 />

# Introduction

This project consists of three modules:

- core: the neural network architecture
- application: plugins, helpers, functionality related to application
- playground: examples with resources
    
# Getting Started

For SBT-Usage, just add this GitHub repository to your dependencies. A maven (or similar) repository is planned for the future.
Also, you may have a look at the playground for some inspiration.

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=487 height=240 />

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

The whole architecture of the net is defined here. For instance, we want to use a sigmoid activation function `fn` for our hidden and output layers. If we would need more layers, we would simply stack them. Optionally, we could provide a `NetSettings` instance to force numeric gradients or disable verbosity.

```scala
Hidden(25, fn) :: Hidden(12, fn) :: Hidden(3, fn) :: Nil
```

Be aware that a default network must start with one `Input` layer and end with one `Output(i, fn)` layer. 

# Training

Let's train our net with the `train` method. It expects the inputs `xs` and their desired outputs `ys`. By design, the type signature is `Seq[Seq[_]]`, because this will promise the most general (Seq, List, Vector, ...) case in scala.
Also, some rates and rules need to be defined, like precision or maximum iterations through a `TrainSettings` instance.

```scala
val (xs, ys) = (-->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0)), -->(->(0.0), ->(1.0), ->(1.0), ->(0.0)))
val trainSets = TrainSettings(stepSize = 2.0, precision = 0.001, maxIterations = 10000)
net.train(xs, ys, trainSets)
```

During training, the derivatives of the net with respect to the weights are constructed, so the optimal weights can be computed. The learning progress will appear on console so we can track it.

# Evaluation

Our trained net can be evaluated with the `evaluate` method.

```scala
net.evaluate(->(0.0, 0.0))
```

# IO

To keep the efforts of a hard, long training phase, it is important to save and load a net.
We can save and load our net to and from file or json-string with `neuroflow.application.plugin.IO`. Scala Pickling is used as the (de-)serialization framework.

```scala
implicit val wp = File.read("/path/to/net.json")
val net = Network(layers)
```

Here, `read` will yield an implicit `WeightProvider` from file. 
(More examples are found within the unit test) 