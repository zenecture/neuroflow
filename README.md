<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

NeuroFlow is a library to construct, sketch, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with <a href="https://github.com/scalanlp/breeze">Breeze</a> on top of <a href="https://github.com/fommil/netlib-java">netlib-java</a> (GPU/CPU).

# Introduction

This project consists of three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources
    
# Getting Started

To use NeuroFlow within your project, add these dependencies (Scala Version 2.12.x):

```scala
libraryDependencies ++= Seq(
  "com.zenecture" %% "neuroflow-core" % "0.805",
  "com.zenecture" %% "neuroflow-application" % "0.805"
)

resolvers ++= Seq("Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/")
```

Seeing code examples is a good way to get started. You may have a look at the playground for some inspiration.
If you want to use neural nets in your project, you can expect a journey full of fun and experiments.

# Net Types

It's hard to hammer a nail using a screw driver, so we have different net types.

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/nettypes.png" width=700 height=250 />

* FFN: A feed forward network. It is good for classification and regression with stationary input. 
* FFN CLUSTER: Use a cluster if you want to represent, compress or cluster your data. Think of word2vec, auto-encoders or principal component analysis. 
* RNN LSTM: A recurrent network. The LSTM model is used here. Use it if you want to do classification and regression with sequential input.

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=443 height=320 />

Let's construct the FFN net depicted above. First, we have to pick the desired behavior:

```scala
import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator._
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
import shapeless._
```

This will give us a fully connected net, which is initialized with random weights in supervised training mode.

```scala
val f = Sigmoid
val net = Network(Input(2) :: Hidden(3, f) :: Output(1, f) :: HNil)
```

The architecture of the net is expressed as a list. We use a sigmoid activation function `f` for our hidden and output layers. 
A more complex net could look like this, with some rates and rules being defined, like precision or maximum iterations, through a `Settings` instance:

```scala
import neuroflow.core.DefaultNetwork._
val (f, g) = (Linear, Sigmoid)
val complexNet = Network(
  Input(50)               ::  
  Cluster(Hidden(10, f))  :: 
  Hidden(20, g)           ::
  Hidden(30, g)           ::
  Hidden(40, g)           :: 
  Output(50, g)           :: HNil, 
  Settings(precision = 1E-5, iterations = 250, 
    learningRate { case iter if iter < 100 => 0.5 case _ => 0.1 },
    regularization = Some(KeepBest), parallelism = 8)
)
```

The learning rate is a partial function from iteration to step size for nets which use gradient descent.

Be aware that a network must start with one `Input(i)` layer and end with one `Output(i, fn)` layer. 
If a network doesn't follow this rule, it won't compile.

# Training

For feed-forward nets, the error function is defined as follows:

    Σ1/2(t - net(x))²

Where `t` is the target vector and `net(x)` the predicted one. The sum `Σ` is taken over all samples and the square `²` gives a convex functional form, which is convenient for gradient descent.

Let's train our `net` with the `train` method. It expects the inputs `xs` and, since it is supervised training, their desired outputs `ys`.
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

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/errgraph2.png" width=494 height=339 />

If you want to be more flexible, e.g. piping the error over the wire to a real-time dashboard, 
you can provide a `closure` of type `Double => Unit` that gets asynchronously executed 
with the respective error as input after each training epoch.

# Evaluation

Our trained net can be evaluated like a regular function:

```scala
val x = ->(0.0, 1.0)
val result = net(x)
// result: Vector(0.980237270455592)
```

The resulting vector has dimension = 1, as specified for our output layer.

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