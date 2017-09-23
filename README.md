<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

NeuroFlow is a library to design, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix and vector operations are performed with <a href="https://github.com/scalanlp/breeze">Breeze</a>.

# Introduction

There are three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources
    
# Getting Started

To use NeuroFlow within your project, add these dependencies (Scala Version 2.12.x):

```scala
libraryDependencies ++= Seq(
  "com.zenecture"   %%   "neuroflow-core"          %   "1.1.5",
  "com.zenecture"   %%   "neuroflow-application"   %   "1.1.5"
)

resolvers ++= Seq("Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/")
```

Seeing code examples is a good way to get started. You may have a look at the playground for some inspiration.
If you want to use neural nets in your project, you can expect a journey full of fun and experiments. 

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=443 height=320 />

Let's construct the fully connected feed-forward net (FFN) depicted above. We have to import everything we need:

```scala
import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator._
import neuroflow.core.WeightProvider.Double.FFN.randomWeights
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import shapeless._
```

This gives a fully connected `DenseNetwork`, which is initialized with random weights by `WeightProvider`.
We import all `Activator` functions so we can place a `Sigmoid` on the layers:

```scala
val (g, h) = (Sigmoid, Sigmoid)
val net = Network(Input(2) :: Dense(3, g) :: Output(1, h) :: HNil)
```

In NeuroFlow, network architectures are expressed as <a href="https://github.com/milessabin/shapeless">HLists</a>. 
They give type-safety and a humble ability to compose groups of layers. For instance, a little deeper net, with some 
rates and rules defined through a `Settings` instance, could look like this:

```scala
val (e, f) = (Linear, Sigmoid)
val bottleNeck =
  Input  (50)               ::
  Focus  (Dense(10, e))     :: HNil
val fullies    =
  Dense  (20,  f)           ::
  Dense  (30,  f)           ::
  Dense  (40,  f)           ::
  Dense  (420, f)           ::
  Dense  (40,  f)           ::
  Dense  (30,  f)           :: 
  Output (20,  f)           :: HNil
val deeperNet = Network(
  bottleNeck ::: fullies, 
  Settings[Double](precision = 1E-5, iterations = 250, 
    learningRate { case (iter, _) if iter < 100 => 1E-4 case (_, _) => 1E-5 },
    regularization = Some(KeepBest), batchSize = Some(8), parallelism = 8
  )
)
```

The learning rate is a partial function from iteration and old learning rate to new learning rate for gradient descent. 
The `batchSize` defines how many samples are presented per weight update and `parallelism` sets the thread pool size, 
since each batch is processed in parallel. Another important aspect is the numerical type of the net, which is set by explicitly annotating
the type `Double` on the settings instance. For instance, on the GPU, you might want to work with `Float` instead of `Double`. 
Have a look at the `Settings` class for the full list of options.

Be aware that a network must start with one `In`-typed layer and end with one `Out`-typed layer. 
If a network doesn't follow this rule, it won't compile.

# Training

Let's train our `net` with the `train` method. It expects the inputs `xs` and, since it is supervised training, their desired outputs `ys`.
In NeuroFlow, you work with Breeze vectors and matrices (`DenseMatrix[Double]`, `DenseVector[Double]` or `Matrix`, `Vector` for brevity). Let's quickly define the training data using the vector notation:

```scala
val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))
net.train(xs, ys) // it's the XOR-Function :-)
```

For feed-forward nets, the error function is defined as follows:

    E(W) = Σ1/2(t - net(x))²

Where `W` are all weights, `t` is the target and `net(x)` the prediction. The sum `Σ` is taken over all samples and 
the square `²` gives a convex functional form, which is convenient for gradient descent.

The training progress will appear on console so we can track it. 
If you want to visualize the error function, you can pipe the errors to a `file` like this:

```scala
  Settings(
    errorFuncOutput = Some(
      ErrorFuncOutput(
        file = Some("~/NF/errorFunc.txt"), 
        action = Some(error => sendToDashboard(error))
      )
    )
  )
```

This way we can use beloved gnuplot to plot the error during training:

```bash
gnuplot> set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.5 
gnuplot> plot '~/NF/errorFunc.txt' with linespoints ls 1
```

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/errgraph2.png" width=494 height=339 />

If you want to be more flexible, e.g. piping the error over the wire to a real-time dashboard, 
you can provide a function closure `action` of type `Double => Unit` which gets executed in the background after each training epoch, 
using the respective error as input.

After work is done, the trained net can be evaluated like a regular function:

```scala
val x = ->(0.0, 1.0)
val result = net(x)
// result: Vector(0.980237270455592)
```

The resulting vector has dimension = 1, as specified for the XOR-example.

# Distributed Training

Let's consider this fully connected FFN:

    Layout: [1200 In, 210 Dense (R), 210 Dense (R), 210 Dense (R), 1200 Out (R)]
    Number of Weights: 592.200 (≈ 4,51813 MB)

On the JVM, a `Double` takes 8 bytes, meaning the derivative of this network requires roughly 4,5 MB per sample. Training with,
let's say, 1 million samples would require ≈ 4,5 TB of RAM for vanilla gradient descent. Luckily, the error function `Σ1/2(t - net(x))²` 
is parallelizable with respect to the sum operator. So, if a single machine offering this amount of memory is not available, 
we can spread the load across several machines instead of batching it.  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/distributedtraining.png" width=800 height=555 />

Distributed gradient descent broadcasts the respective weight updates between the training epochs to all nodes. 
In our example, the overhead is 2*4,5=9 MB network traffic per node and iteration, while gaining computational parallelism.

```scala
import neuroflow.nets.distributed.DenseNetwork._

object Coordinator extends App {

  val nodes = Set(Node("localhost", 2553) /* ... */)

  def coordinator = {
    val f = ReLU
    val net =
      Network(
        Input (1200) :: Dense(210, f) :: Dense(210, f) :: Dense(210, f) :: Output(1200, f) :: HNil,
        Settings[Double](
          coordinator  = Node("localhost", 2552),
          transport    = Transport(messageGroupSize = 100000, frameSize = "128 MiB")
        )
      )
    net.train(nodes)
  }

}
```

The network is defined in the `Coordinator`. The `train` method will trigger training for all `nodes`. 

```scala
import neuroflow.nets.distributed.DenseExecutor

object Executor extends App {

  val (xs, ys) =  (???, ???) // Local Training Data
  DenseExecutor(Node("localhost", 2553), xs, ys)

}
```

The `Executor`, a single node, loads the local data source, boots the networking subsystem and listens for incoming jobs.

# IO

Using `neuroflow.application.plugin.IO` we can store the weights represented as JSON strings. Look at this:

```scala
val file = "/path/to/net.nf"
implicit val weightProvider = IO.File.readDouble(file)
val net = Network(layers)
// ... do work.
IO.File.write(net.weights, file)
```

Here, `IO.File.read` will yield an implicit `WeightProvider` from file to construct a net.
The weights will be saved to the same file with `IO.File.write`. 
If the desired target is a database, simply use `IO.Json.write` instead and save it as a raw JSON string. 
However, all important types extend `Serializable`, so feel free to work with the bytes on your own.