<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/newlogo.png" width=480 height=132 />

NeuroFlow is a library to design, train and evaluate Artificial Neural Networks.

# Getting Started

There are three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources

To use NeuroFlow, add these dependencies (Scala Version 2.12.x, oss.sonatype.org) to your SBT project:

```scala
libraryDependencies ++= Seq(
  "com.zenecture"   %%   "neuroflow-core"          %   "1.3.6",
  "com.zenecture"   %%   "neuroflow-application"   %   "1.3.6"
)
```

If you are new to Neural Nets, you can read about the core principles here:

  - <a href="http://www.znctr.com/blog/artificial-neural-networks">znctr.com/blog/artificial-neural-networks</a>
  
Seeing code examples is also a good way to get started. You may have a look at the playground for some basic inspiration.

Neural Nets bring a lot of joy in your project, a journey full of fun and experiments. 

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

The `lossFunction` computes loss and gradient, which is backpropped into the raw output layer of the net. The `updateRule` defines how weights are updated for gradient descent. The `batchSize` defines 
how many samples are presented per weight update. The `learningRate` is a partial function from current iteration and learning rate producing a new learning rate. Training terminates after `iterations`, or if loss 
satisfies `precision`. 

Another important aspect is the numerical type of the net, which is set by explicitly annotating `Double` on the settings instance.  For instance, on the GPU, you might want to work 
with `Float` instead. Have a look at the `Settings` class for the complete list of options.

Be aware that a network must start with one `In`-typed layer and end with one `Out`-typed layer. 
If a network doesn't follow this rule, it won't compile.

# Training

We want to map from a two-dimensional vector `x` to a one-dimensional vector `y` with our architecture.
There are many functions out there of this kind; here we use the XOR-Function. It is linearily not separable,
so we can check whether our net can capture this non-linearity.

In NeuroFlow, you work with <a href="https://github.com/scalanlp/breeze">Breeze</a>, in particular with `DenseVector[V]` and `DenseMatrix[V]`. 
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
And then we can `train` our `net`. The `SquaredMeanError` loss function is defined as follows:

    L(W) = Σ1/2(t - net(x))²

Where `W` are the weights, `t` is the target and `net(x)` the prediction. The sum `Σ` is taken over all samples and 
the square `²` gives a convex functional form. For 1-of-K classification, there is also the <a href="http://www.znctr.com/blog/digit-recognition#softmax">`Softmax`</a> loss function, 
but here we treat the XOR-adder as a regression challenge.

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/derivative.png" width=443 height=320 />

<small><em>Example: Derivative for w<sub>8</sub></em></small>

The training progress is printed on console so we can track it.

### Recommendation

Here is a `Settings` recommendation for running long sessions in a convenient way.

```scala
  Settings(
    lossFuncOutput = Some(LossFuncOutput(file = Some("~/NF/lossFunc.txt"), action = Some(loss => sendToDashboard(loss)))),
    waypoint       = Some(Waypoint(nth = 3, (iter, weights) => IO.File.write(weights, s"weights-iter-$iter.nf")))
  )
```

To visualize the loss function, we can append the loss of a training step to `file` with `LossFuncOutput`.
Now we can use beloved gnuplot:

```bash
gnuplot> set style line 1 lc rgb '#0060ad' lt 1 lw 1 pt 7 ps 0.5 
gnuplot> plot '~/NF/lossFunc.txt' with linespoints ls 1
```

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/errgraph3.png" width=448 height=321 />

To be more flexible, we can provide function `action` of type `Double => Unit` which gets executed in the background 
after each training step, using the respective loss as input. One example is sending the loss to a real-time TV dashboard.

It is a good idea to make use of a `Waypoint[V]` for long sessions, since they can be difficult, e. g. running on not always stable cloud instances, or to backup expensive iterations.
Every `nth` steps, the specified function is executed, receiving as input the iteration count and a snapshot of the weights.

# Evaluation

When training is done, the net can be evaluated like a regular function:

```scala
val x = ->(0.0, 1.0)
val result = net(x)
// result: DenseVector(0.980237270455592)
```

The resulting vector has dimension = 1, as specified for the XOR-example.

# Using GPU

If your graphics card supports nVidia's <a href="https://developer.nvidia.com/cuda-gpus">CUDA</a> (Compute Capability >= 3.0), you can train nets on the GPU, 
which is recommended for large nets with millions of weights and samples. On the contrary, smaller nets are faster to train on CPU, because while NeuroFlow 
is busy copying batches between host and GPU, CPU is already done. 

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

# Persistence

With `neuroflow.application.plugin.IO`, we can save and load the weights of a network. The weights are encoded in JSON format.

```scala
val file = "/path/to/net.nf"
implicit val weightProvider = IO.File.readDouble(file)
val net = Network(layers, settings)
// training ...
IO.File.write(net.weights, file)
```

The implicit `WeightProvider[Double]` to construct `net` comes from `IO.File.readDouble`.
To save the weights back to `file`, we use `IO.File.write`. To write into a database instead, 
we can use `IO.Json.write` to retrieve a raw JSON string and fire a SQL query with it.
