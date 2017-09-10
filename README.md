<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

NeuroFlow is a library to train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with <a href="https://github.com/scalanlp/breeze">Breeze</a> on top of <a href="https://github.com/fommil/netlib-java">netlib-java</a> (GPU/CPU).
Large training sets can be distributed over physical nodes and trained in parallel, using <a href="https://github.com/akka/akka">Akka</a> (UDP) for inter-node communication.

# Introduction

This project consists of three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources
    
# Getting Started

To use NeuroFlow within your project, add these dependencies (Scala Version 2.12.x):

```scala
libraryDependencies ++= Seq(
  "com.zenecture" %% "neuroflow-core" % "1.00.0",
  "com.zenecture" %% "neuroflow-application" % "1.00.0"
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
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
import shapeless._
```

This gives us a fully connected net, which is initialized with random weights in supervised training mode. 

```scala
val (g, h) = (Sigmoid, Sigmoid)
val net = Network(Input(2) :: Dense(3, g) :: Output(1, h) :: HNil)
```

The architecture of the net is expressed as a list. We use sigmoid activation functions `g` and `h` for hidden and output layers. 
A little deeper net, with some rates and rules defined, like precision or maximum iterations, through a `Settings` instance, 
could look like this:

```scala
val (e, f) = (Linear, Sigmoid)
val deeperNet = Network(
  Input(50)               ::  
  Focus(Dense(10, e))     :: 
  Dense(20, f)            ::
  Dense(30, f)            ::
  Dense(40, f)            :: 
  Output(50, f)           :: HNil, 
  Settings(precision = 1E-5, iterations = 250, 
    learningRate { case iter if iter < 100 => 1E-4 case _ => 1E-5 },
    regularization = Some(KeepBest), parallelism = 8)
)
```

The learning rate is a partial function from iteration to step size for nets which use gradient descent.

Be aware that a network must start with one `In`-typed layer and end with one `Out`-typed layer. 
If a network doesn't follow this rule, it won't compile.

# Training

For feed-forward nets, the error function is defined as follows:

    Σ1/2(t - net(x))²

Where `t` is the target and `net(x)` the prediction. The sum `Σ` is taken over all samples and the square `²` gives a convex functional form, which is convenient for gradient descent.

Let's train our `net` with the `train` method. It expects the inputs `xs` and, since it is supervised training, their desired outputs `ys`.
For our little example, let's quickly define the training data using the vector notation:

```scala
val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))
net.train(xs, ys) // it's the XOR-Function :-)
```

The training progress will appear on console so we can track it. 
If you want to visualize the error function graph during training, 
you can pipe the errors to any `file` like this:

```scala
  Settings(
    errorFuncOutput = Some(ErrorFuncOutput(
      file = Some("~/NF/errorFunc.txt"), 
      closure = Some(error => proceed(error))))
  )
```

We can use beloved gnuplot to come up with a nice plot of our error function over time:

```bash
gnuplot> set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.5 
gnuplot> plot '~/NF/errorFunc.txt' with linespoints ls 1
```

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/errgraph2.png" width=494 height=339 />

If you want to be more flexible, e.g. piping the error over the wire to a real-time dashboard, 
you can provide a function `closure` of type `Double => Unit` which gets executed in the background after each training epoch, 
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

    Layout: [1200 In, 210 Hidden (ReLU), 210 Hidden (ReLU), 210 Hidden (ReLU), 1200 Out (ReLU)]
    Number of Weights: 592.200 (≈ 4,51813 MB)

On the JVM, a `Double` takes 8 bytes, meaning the derivative of this network requires roughly 4,5 MB per sample. Training with,
let's say, 1 million samples would require ≈ 4,5 TB of RAM for gradient descent. Luckily, the error function `Σ1/2(t - net(x))²` 
is parallelizable with respect to the sum operator. So, if a single machine offering this amount of memory is not available, 
we can spread the load across several machines instead of batching it.  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/distributedtraining.png" width=800 height=555 />

Distributed gradient descent broadcasts the respective weight updates between the training epochs to all nodes. 
In our example, the overhead is 2*4,5=9 MB network traffic per node and iteration, while gaining computational parallelism.

```scala
import neuroflow.nets.distributed.DefaultNetwork._

object Coordinator extends App {

  val nodes = Set(Node("localhost", 2553) /* ... */)

  def coordinator = {
    val f   = ReLU
    val net =
      Network(
        Input (1200) :: Hidden(210, f) :: Hidden(210, f) :: Hidden(210, f) :: Output(1200, f) :: HNil,
        Settings(
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
import neuroflow.nets.distributed.DefaultExecutor

object Executor extends App {

  val (xs, ys) =  (???, ???) // Local Training Data
  DefaultExecutor(Node("localhost", 2553), xs, ys)

}
```

The `Executor`, a single node, loads the local data source, boots the networking subsystem and listens for incoming jobs.

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