# NeuroFlow

NeuroFlow is a lightweight library to construct, train and evaluate Artificial Neural Networks.
It is written in Scala, matrix operations are performed with Breeze (+ NetLib for near-native performance).
Type-safety, when needed, comes from Shapeless.

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/logo.png" width=471 height=126 />

# Introduction

This project consists of three modules:

- core: the building blocks to create neural network architectures
- application: plugins, helpers, functionality related to various applications
- playground: examples with resources
    
# Getting Started

To use Neuroflow within your project, add these dependencies (Scala Version 2.11.x):

```scala
libraryDependencies ++= Seq(
  "com.zenecture" % "neuroflow-core_2.11" % "0.200-SNAPSHOT",
  "com.zenecture" % "neuroflow-application_2.11" % "0.200-SNAPSHOT"
)
```

Usually the Sonatype repository resolvers are provided by default. However, sometimes the explicit definition is needed:

```scala
resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
```

Seeing some code examples is a good way to get started. 
You may have a look at the playground for some inspiration.

If you want to use neural nets in your project, you can expect a journey full of experiments.

Never forget, each challenge is unique.

# Construction of a Net  

<img src="https://raw.githubusercontent.com/zenecture/zenecture-docs/master/neuroflow/arch.png" width=443 height=320 />

Let's construct the net depicted above. First, we have to pick the desired behavior:

```scala
import neuroflow.nets.DefaultNetwork._
import neuroflow.core._
import neuroflow.core.WeightProvider.randomWeights // Default, if not imported explicitly like here.
import neuroflow.application.plugin.Style._
import shapeless._
```

The first import gives us an implicit constructor for the default net implementation with gradient descent. 
The second one yields a weight provider, which determines the initial weight values. The last one is eye candy to keep the notation a little shorter. 
The idea behind this 'import a-la-carte' is to change the underlying net behavior without changing the 'meat' of the code.

```scala
val fn = Sigmoid.apply
val net = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: HNil)
```

The whole architecture of the net is defined here. We want to use a sigmoid activation function `fn` for our hidden and output layers. 
Also, some rates and rules need to be defined, like precision or maximum iterations through a `Settings` instance. If we would need a more complex net, we would simply stack layers and functions:

```scala
val fn = Sigmoid.apply
val gn = Tanh.apply
val settings = Settings(verbose = true, learningRate = 0.001, precision = 0.001, maxIterations = 200, regularization = None, approximation = None, specifics = None)
val net = Network(Input(50) :: Hidden(20, fn) :: Hidden(10, gn) :: Output(2, fn) :: HNil, settings)
```

Be aware that a network must start with one `Input(i)` layer and end with one `Output(i, fn)` layer. If a network doesn't follow this rule, it won't compile. 

# Training

Let's train our net with the `train` method. It expects the inputs `xs` and their desired outputs `ys`. By design, the type signature of `train` is `Seq[Seq[_]]`, because this promises the most general (Seq, List, Vector, ...) case in Scala.

```scala
val xs = -->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
val ys = -->(->(0.0), ->(1.0), ->(1.0), ->(0.0))
net.train(xs, ys)
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

# Todo

- Implement LSTM and softmax blocks
- Investigate usefulness of known regularization techniques
- Provide helpers for easy parallelization (Akka? Spark? scala.collection.parallel?)
- Check whether GPU based matrix/net implementations are worth the hassle. Intel MKL CPU was faster than CUDA/OpenCL on my MacBookPro with a Geforce GT750 for SGEMM? The netlib benchmark confirms my observation: [github.com/fommil/netlib-java](https://github.com/fommil/netlib-java)