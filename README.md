# NeuroFlow

NeuroFlow is a lightweight library to construct, sketch, train and evaluate Artificial Neural Networks (FFN, RNN).
It is written in Scala, matrix operations are performed with Breeze (+ NetLib).

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

# Introduction

This project consists of three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources
    
# Getting Started

To use Neuroflow within your project, add these dependencies (Scala Version 2.11.8, 2.12.1):

```scala
libraryDependencies ++= Seq(
  "com.zenecture" %% "neuroflow-core" % "0.500",
  "com.zenecture" %% "neuroflow-application" % "0.500"
)
```

Usually the Sonatype repository resolvers are provided by default. However, sometimes the explicit definition is needed:

```scala
resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
```

Seeing code examples is a good way to get started. You may have a look at the playground for some inspiration.
If you want to use neural nets in your project, you can expect a journey full of fun and experiments.

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=443 height=320 />

Let's construct the net depicted above. First, we have to pick the desired behavior:

```scala
import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator._
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
import shapeless._
```

This will give us a fully connected ANN, with random weights and standard gradient descent / backprop training.


```scala
val fn = Sigmoid
val net = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: HNil)
```

The architecture of the net is defined here. We use a sigmoid activation function `fn` for our hidden and output layers. 
Also, some rates and rules need to be defined, like precision or maximum iterations through a `Settings` instance. 

If we would need a more complex net, we would simply stack layers and functions:

```scala
val f = Sigmoid
val g = Tanh
val settings = Settings(verbose = true, learningRate = 0.01, precision = 0.001, maxIterations = 200)
val net = Network(Input(50) :: Hidden(20, f) :: Hidden(10, g) :: Output(2, f) :: HNil, settings)
```

Be aware that a network must start with one `Input(i)` layer and end with one `Output(i, fn)` layer. 
If a network doesn't follow this rule, it won't compile.

# Training

Let's train our net with the `train` method. It expects the inputs `xs` and their desired outputs `ys`. 
By design, the type signature of `train` is `Seq[Seq[_]]`, since this promises the most general case when working with external data. 
For our little example, let's simply define the training data using the vector notation:

```scala
val xs = -->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = -->(->(0.0), ->(1.0), ->(1.0), ->(0.0))
net.train(xs, ys)
```

The training progress will appear on console so we can track it. 
If you want to visualize the error function graph during training, 
you can pipe the `ErrorFuncOutput` to any `file` like this:

```scala
    Settings(
      errorFuncOutput = Some(ErrorFuncOutput(
        file = Some("~/NF/errorFunc.txt"), 
        closure = Some(error => proceed(error))))
    )
```

Let's use beloved gnuplot to come up with a plot of our error function over time:

```bash
gnuplot> set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.5 
gnuplot> plot '~/NF/errorFunc.txt' with linespoints ls 1
```

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/errgraph.png" width=400 height=400 />

If you want to be more flexible, e.g. piping the error over the wire to a real-time dashboard, 
you can provide a `closure` of type `Double => Unit` that gets asynchronously executed 
with the respective error as input after each training epoch.

# Evaluation

Our trained net can be evaluated with the `evaluate` method.

```scala
val result = net.evaluate(->(0.0, 0.0))
```

This will give us a result vector (kind `Seq[_]`) with the dimension of our specified output layer.

# IO

In the component `neuroflow.application.plugin.IO` we will find functionality to save and load nets, especially weights.
Scala Pickling is used as the (de-)serialization framework. Example:

```scala
val file = "/path/to/net.nf"
implicit val wp = File.read(file)
val net = Network(layers)
File.write(net, file)
```

Here, `File.read` will yield an implicit `WeightProvider` from file to construct a net. 
A net is always constructed like this. Instead of initializing it with random weights, 
we just load them from file (or json). Afterwards it will be saved to the same file with `File.write`. 
If the desired target is a database, simply use `Json.write` instead and save it on string-level. 
However, all important types extend `Serializable`, so feel free to work with the bytes on your own.