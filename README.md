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

To use NeuroFlow within your project, add these dependencies (Scala Version 2.11.8, 2.12.1):

```scala
libraryDependencies ++= Seq(
  "com.zenecture" %% "neuroflow-core" % "0.500",
  "com.zenecture" %% "neuroflow-application" % "0.500"
)

resolvers ++= Seq("Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/")
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

This will give us a fully connected ANN, which is initialized with random weights and supervised training mode.

```scala
val fn = Sigmoid
val net = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: HNil)
```

This is the most basic net. The architecture of the net is a list. We use a sigmoid activation function `fn` for our hidden and output layers. 
A more complex net could look like this, with some rates and rules being defined, like precision or maximum iterations, through a `Settings` instance:

```scala
val s = Sigmoid
val x = Linear
val settings = Settings(learningRate = 0.01, precision = 1E-5, iterations = 200)
val net = Network(Input(50) :: Hidden(20, s) :: Cluster(10, x) :: Hidden(20, s) :: Output(50, s) :: HNil, settings)
```

Be aware that a network must start with one `Input(i)` layer and end with one `Output(i, fn)` layer. 
If a network doesn't follow this rule, it won't compile.

# Training

Let's train our net with the `train` method. It expects the inputs `xs` and, since it is supervised training, their desired outputs `ys`.
For our little example, let's quickly define the training data using the vector notation:

```scala
val xs = -->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = -->(->(0.0), ->(1.0), ->(1.0), ->(0.0))
net.train(xs, ys) // it's the XOR-Function :-)
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
val result = net.evaluate(->(0.0, 1.0))
// result: Vector(0.9785958704533262)
```

This will give us a result vector with the dimension of our specified output layer.

# IO

Using `neuroflow.application.plugin.IO` we can store the weights represented as JSON strings. Look at this:

```scala
val file = "/path/to/net.nf"
implicit val wp = IO.File.read(file)
val net = Network(layers)
IO.File.write(net, file)
```

Here, `IO.File.read` will yield an implicit `WeightProvider` from file to construct a net.
Afterwards, the weights will be saved to the same file with `IO.File.write`. 
If the desired target is a database, simply use `IO.Json.write` instead and save it as a raw JSON string. 
However, all important types extend `Serializable`, so feel free to work with the bytes on your own.