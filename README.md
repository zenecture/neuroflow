<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

NeuroFlow is a library to design, train and evaluate Artificial Neural Networks.

# Getting Started

There are three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources

To use NeuroFlow within your project, add these dependencies (Scala Version 2.12.x, oss.sonatype.org):

```scala
libraryDependencies ++= Seq(
  "com.zenecture"   %%   "neuroflow-core"          %   "1.3.5",
  "com.zenecture"   %%   "neuroflow-application"   %   "1.3.5"
)
```

If you are new to Neural Nets, you can read about the core principles here:

  - <a href="http://www.znctr.com/blog/artificial-neural-networks">znctr.com/blog/artificial-neural-networks</a>
  
Seeing code examples is also a good way to get started. You may have a look at the playground for some basic inspiration.
If you want to use neural nets in your project, you can expect a journey full of fun and experiments. 

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=443 height=320 />

Let's construct the fully connected feed-forward net (FFN) depicted above.

```scala
import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import shapeless._

implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

val (g, h) = (Sigmoid, Sigmoid)
val net = Network(Input(2) :: Dense(3, g) :: Output(1, h) :: HNil)
```

This gives a fully connected `DenseNetwork`, which is initialized with random weights in range (-1, 1) by `WeightProvider`.
Further, we have pre-defined activators, so we can place a `Sigmoid` on the layers.

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
  Settings[Double](
    lossFunction = SquaredMeanError(), updateRule = Vanilla(), batchSize = Some(8), iterations = 250,
    learningRate { case (iter, _) if iter < 100 => 1E-4 case (_, _) => 1E-5 }, precision = 1E-5
  )
)
```

The `lossFunction` computes loss and gradient, which will be backpropped into the raw output layer of a net. The `updateRule` defines how weights are updated for gradient descent. The `batchSize` defines 
how many samples are presented per weight update. The learning rate is a partial function from current iteration and learning rate to new learning rate. Training terminates after `iterations`, or if loss 
satisfies `precision`. 

Another important aspect is the numerical type of the net, which is set by explicitly annotating `Double` on the settings instance.  For instance, on the GPU, you might want to work 
with `Float` instead of `Double`. Have a look at the `Settings` class for the complete list of options.

Be aware that a network must start with one `In`-typed layer and end with one `Out`-typed layer. 
If a network doesn't follow this rule, it won't compile.

# Training

We want to map from a two-dimensional vector `x` to a one-dimensional vector `y` with our architecture.
There are many functions out there of this kind; here we use the XOR-Function. It is linearily not separable,
so we can check whether our net can capture this non-linearity.

In NeuroFlow, you work with <a href="https://github.com/scalanlp/breeze">Breeze</a> vectors and matrices (`DenseMatrix[V]`, `DenseVector[V]`). 
To define the training data we use the built-in vector notation:

```scala
val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))

/*
  It's the XOR-Function :-).
  Or: the net learns to add binary digits modulo 2.
*/

net.train(xs, ys)
```
And then we can `train` our `net`. For our XOR-feed-forward net, the loss function is defined as follows:

    L(W) = Σ1/2(t - net(x))²

Where `W` are the weights, `t` is the target and `net(x)` the prediction. The sum `Σ` is taken over all samples and 
the square `²` gives a convex functional form, which is convenient for gradient descent.

The training progress will appear on console so we can track it. 
If you want to visualize the loss function, you can pipe the values to a `file` like this:

```scala
  Settings(
    lossFuncOutput = Some(
      LossFuncOutput(
        file = Some("~/NF/lossFunc.txt"), 
        action = Some(loss => sendToDashboard(loss))
      )
    )
  )
```

This way we can use beloved gnuplot to plot the loss during training:

```bash
gnuplot> set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.5 
gnuplot> plot '~/NF/lossFunc.txt' with linespoints ls 1
```

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/errgraph3.png" width=448 height=321 />

If you want to be more flexible, e.g. piping the loss over the wire to a real-time dashboard, 
you can provide a function closure `action` of type `Double => Unit` which gets executed in the background after each training epoch, 
using the respective loss as input.

After work is done, the trained net can be evaluated like a regular function:

```scala
val x = ->(0.0, 1.0)
val result = net(x)
// result: Vector(0.980237270455592)
```

The resulting vector has dimension = 1, as specified for the XOR-example.

# Using GPU

If your graphics card supports nVidia's <a href="https://developer.nvidia.com/cuda-gpus">CUDA</a> (Compute Capability >= 3.0), you can train nets on the GPU, which is recommended for large nets with millions of weights.
On the contrary, smaller nets are faster to train on CPU, because while NeuroFlow is busy copying batches between host and GPU, CPU is already done. 

To enable the GPU for NeuroFlow, you have to install the CUDA driver and toolkit (0.8.x). Example for Linux (Ubuntu 16.04):

```bash
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install cuda-toolkit-8-0
``` 

With both driver and toolkit installed, add the <a href="http://jcuda.org">jCUDA</a> dependencies to your SBT project:

```scala
resolvers ++= Seq(
  "neuroflow-libs" at "https://github.com/zenecture/neuroflow-libs/raw/master/"
)
```

Then, you can import a GPU implementation for your model:

```scala
import neuroflow.nets.gpu.DenseNetwork._
```

# Distributed Training

Let's consider this fully connected FFN:

    Layout: [1200 In, 210 Dense (R), 210 Dense (R), 210 Dense (R), 1200 Out (R)]
    Number of Weights: 592.200 (≈ 4,51813 MB)

On the JVM, a `Double` takes 8 bytes, meaning the derivative of this network requires roughly 4,5 MB per sample. Training with,
let's say, 1 million samples would require ≈ 4,5 TB of RAM for vanilla gradient descent. Luckily, the loss function `Σ1/2(t - net(x))²` 
is parallelizable with respect to the sum operator. So, if a single machine offering this amount of memory is not available, 
we can spread the load across several machines instead of batching it.  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/distributedtraining.png" width=800 height=555 />

Distributed gradient descent broadcasts the respective weight updates between the training epochs to all nodes. 
In our example, the overhead is 2*4,5=9 MB network traffic per node and iteration, while gaining computational parallelism

<em>However, note that on-line or mini-batch training can have much faster convergence (depending on the level of redundancy within the data) 
than a full distributed batch, even when using several machines.</em>

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

# Saving, Loading a Model

With `neuroflow.application.plugin.IO`, we can save and load the weights of a network represented as a JSON string. Look at this:

```scala
val file = "/path/to/net.nf"
implicit val weightProvider = IO.File.readDouble(file)
val net = Network(layers)
// training ...
IO.File.write(net.weights, file)
```

Here, `IO.File.read` yields an implicit `WeightProvider` from file to construct a net.
After training is done, the weights can be saved back with `IO.File.write`. If the desired target is a database, 
you could use `IO.Json.write` to retrieve a raw JSON string and then fire a SQL query with it.
