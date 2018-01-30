package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common._
import neuroflow.core.Network._

import scala.annotation.implicitNotFound
import scala.collection._

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network extends TypeAliases {

  /**
    * Constructs a new [[Network]] with the respective [[Constructor]] in scope.
    * Additionally, it proves that the [[Layout]] graph is a valid composition.
    */
  def apply[V, L <: Layout, N <: Network[_, _, _]]
                                     (layout: L,
                                      settings: Settings[V] = Settings[V]())
                                     (implicit
                                      startsWith: L StartsWith In,
                                      endsWith: L EndsWith Out,
                                      weightProvider: WeightProvider[V],
                                      constructor: Constructor[V, N],
                                      extractor: Extractor[L, Layer, V]): N = {
    val (layers, loss) = extractor(layout)
    constructor(layers, loss, settings)
  }

}


trait TypeAliases {

  type Vectors[V]    =    Seq[DenseVector[V]]
  type Matrices[V]   =   Seq[DenseMatrix[V]]
  type Weights[V]    =    IndexedSeq[DenseMatrix[V]]
  type LearningRate  =  PartialFunction[(Int, Double), Double]

}


/** A minimal constructor for a [[Network]]. */
@implicitNotFound("No `Constructor` in scope. Import your desired network or try: import neuroflow.nets.cpu.DenseNetwork._")
trait Constructor[V, +N <: Network[_, _, _]] {
  def apply(ls: Seq[Layer], loss: LossFunction[V], settings: Settings[V])(implicit weightProvider: WeightProvider[V]): N
}


/**
  *
  * Settings of a neural network, where:
  *
  *   `verbose`             Indicates logging behavior on console.
  *   `learningRate`        A function from current iteration and learning rate, producing a new learning rate.
  *   `updateRule`          Defines the relationship between gradient, weights and learning rate during training.
  *   `precision`           The training will stop if precision is high enough.
  *   `iterations`          The training will stop if maximum iterations is reached.
  *   `prettyPrint`         If true, the layout is rendered graphically on console.
  *   `coordinator`         The coordinator host address for distributed training.
  *   `transport`           Transport throughput specifics for distributed training.
  *   `parallelism`         Controls how many threads are used for distributed training.
  *   `batchSize`           Controls how many samples are presented per weight update. (1=on-line, ..., n=full-batch)
  *   `lossFuncOutput`      Prints the loss to the specified file/closure.
  *   `waypoint`            Periodic actions can be executed, e.g. saving the weights every n steps.
  *   `approximation`       If set, the gradients are approximated numerically.
  *   `regularization`      The respective regulator tries to avoid over-fitting.
  *   `partitions`          A sequential training sequence can be partitioned for RNNs. (0 index-based)
  *   `specifics`           Some nets use specific parameters set in the `specifics` map.
  *
  */
case class Settings[V]
                   (verbose           :  Boolean                      =  true,
                    learningRate      :  LearningRate                 =  { case (_, _) => 1E-4 },
                    updateRule        :  Update[V]                    =  Vanilla[V](),
                    precision         :  Double                       =  1E-5,
                    iterations        :  Int                          =  100,
                    prettyPrint       :  Boolean                      =  true,
                    coordinator       :  Node                         =  Node("localhost", 2552),
                    transport         :  Transport                    =  Transport(100000, "128 MiB"),
                    parallelism       :  Option[Int]                  =  Some(Runtime.getRuntime.availableProcessors),
                    batchSize         :  Option[Int]                  =  None,
                    lossFuncOutput    :  Option[LossFuncOutput]       =  None,
                    waypoint          :  Option[Waypoint[V]]          =  None,
                    approximation     :  Option[Approximation[V]]     =  None,
                    regularization    :  Option[Regularization]       =  None,
                    partitions        :  Option[Set[Int]]             =  None,
                    specifics         :  Option[Map[String, Double]]  =  None) extends Serializable


trait IllusionBreaker { self: Network[_, _, _] =>

  /**
    * Checks if the [[Settings]] are properly defined for this network.
    * Throws a [[neuroflow.core.IllusionBreaker.SettingsNotSupportedException]] if not. Default behavior is no op.
    */
  def checkSettings(): Unit = ()

}


object IllusionBreaker {

  class SettingsNotSupportedException(message: String) extends Exception(message)
  class NotSoundException(message: String) extends Exception(message)

}


trait Network[V, In, Out] extends (In => Out) with Logs with LossFuncGrapher with IllusionBreaker with Welcoming with Serializable {

  checkSettings()

  sayHi()

  val identifier: String

  val numericPrecision: String

  /** Layers of this neural network. */
  val layers: Seq[Layer]

  /** The attached loss function. */
  val lossFunction: LossFunction[V]

  /** Settings of this neural network. */
  val settings: Settings[V]

  /** The weights are a bunch of matrices. */
  val weights: Weights[V]

  /**
    * Computes output for given input `in`.
    * Alias for `net(x)` syntax.
    */
  def evaluate(in: In): Out = apply(in)

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


trait FFN[V] extends Network[V, DenseVector[V], DenseVector[V]] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors[V], ys: Vectors[V]): Unit

}


trait CNN[V] extends Network[V, DenseMatrix[V], DenseVector[V]] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Matrices[V], ys: Vectors[V]): Unit

}


trait RNN[V] extends Network[V, Vectors[V], Vectors[V]] {

  /**
    * Takes input `xs` and trains this network against output `ys`.
    */
  def train(xs: Vectors[V], ys: Vectors[V]): Unit

}


trait DistributedTraining {

  /**
    * Triggers execution of training for nodes `ns`.
    */
  def train(ns: collection.Set[Node]): Unit

}


trait DistFFN[V] extends Network[V, DenseVector[V], DenseVector[V]] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}


trait DistCNN[V] extends Network[V, DenseMatrix[V], DenseVector[V]] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}
