package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.math.Field
import neuroflow.common._
import neuroflow.core.Network._
import shapeless._
import shapeless.ops.hlist._

import scala.annotation.implicitNotFound
import scala.collection._

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network extends TypeAliases {

  /**
    * Constructs a new [[Network]] with the respective [[Constructor]] in scope.
    * Additionally, it will prove that the architecture of the net is sound.
    */
  def apply[V, T <: Network[_, _, _], L <: HList]
                                     (layers: L,
                                      settings: Settings[V] = Settings[V]())
                                     (implicit
                                      startsWith: L StartsWith In,
                                      endsWith: L EndsWith Out,
                                      weightProvider: WeightProvider[V],
                                      constructor: Constructor[V, T],
                                      toList: L ToList Layer): T = {
    constructor(layers.toList, settings)
  }

}


/** For the sake of beauty. */
trait TypeAliases {

  type SVector[V]   =  scala.Vector[V]
  type Vector[V]    =  DenseVector[V]
  type Matrix[V]    =  DenseMatrix[V]
  type Vectors[V]   =  Seq[Vector[V]]
  type Matrices[V]  =  Seq[Matrix[V]]
  type Weights[V]   =  IndexedSeq[Matrix[V]]
  type Learning     =  PartialFunction[(Int, Double), Double]

}


/** A minimal constructor for a [[Network]]. */
@implicitNotFound("No `Constructor` in scope. Import your desired network or try: import neuroflow.nets.cpu.DenseNetwork._")
trait Constructor[V, +T <: Network[_, _, _]] {
  def apply(ls: Seq[Layer], settings: Settings[V])(implicit weightProvider: WeightProvider[V]): T
}


/**
  *
  * Settings of a neural network, where:
  *
  *   `verbose`             Indicates logging behavior on console.
  *   `lossFunction`        The loss function used during training.
  *   `learningRate`        A function from current iteration and learning rate, producing a new learning rate.
  *   `updateRule`          Defines the relationship between gradient, weights and learning rate during training.
  *   `precision`           The training will stop if precision is high enough
  *   `iterations`          The training will stop if maximum iterations is reached.
  *   `prettyPrint`         If true, the layout will be rendered graphically on console.
  *   `coordinator`         For distributed training, the coordinator host needs to be set
  *   `transport`           For distributed training, the transport details need to be set
  *   `parallelism`         Controls how many threads will be used for training.
  *   `batchSize`           Controls how many samples are presented per weight update. (1=on-line, ..., n=full-batch)
  *   `lossFuncOutput`      Prints the loss to the specified file/closure.
  *   `regularization`      The respective regulator will try to avoid over-fitting.
  *   `waypoint`            A waypoint action can be specified, e.g. saving the weights along the way.
  *   `approximation`       If true, the gradients will be approximated numerically.
  *   `partitions`          A sequential training sequence can be partitioned for RNNs. (0 index-based)
  *   `specifics`           Some nets use specific parameters set in the `specifics` map.
  *
  */
case class Settings[V]
                   (verbose           :  Boolean                      =  true,
                    lossFunction      :  Loss[V]                      =  SquaredMeanError[V],
                    learningRate      :  Learning                     =  { case (_, _) => 1E-4 },
                    updateRule        :  Update[V]                    =  Vanilla[V],
                    precision         :  Double                       =  1E-5,
                    iterations        :  Int                          =  100,
                    prettyPrint       :  Boolean                      =  false,
                    coordinator       :  Node                         =  Node("localhost", 2552),
                    transport         :  Transport                    =  Transport(100000, "128 MiB"),
                    parallelism       :  Option[Int]                  =  Some(Runtime.getRuntime.availableProcessors),
                    batchSize         :  Option[Int]                  =  None,
                    lossFuncOutput    :  Option[LossFuncOutput]       =  None,
                    regularization    :  Option[Regularization]       =  None,
                    waypoint          :  Option[Waypoint[V]]          =  None,
                    approximation     :  Option[Approximation]        =  None,
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

  val numericPrecision: String

  val identifier: String

  /** Settings of this neural network. */
  val settings: Settings[V]

  /** Layers of this neural network. */
  val layers: Seq[Layer]

  /** The weights are a bunch of matrices. */
  val weights: Weights[V]

  /**
    * Computes output for given input `in`.
    * Alias for `net(x)` syntax.
    */
  def evaluate(in: In): Out = apply(in)

  override def toString: String = weights.foldLeft("")(_ + "\n---\n" + _)

}


trait FFN[V] extends Network[V, Vector[V], Vector[V]] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors[V], ys: Vectors[V]): Unit

}


trait CNN[V] extends Network[V, Matrices[V], Vector[V]] {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Seq[Matrices[V]], ys: Vectors[V]): Unit

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


trait DistFFN[V] extends Network[V, Vector[V], Vector[V]] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("FFNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}


trait DistCNN[V] extends Network[V, Matrices[V], Vector[V]] with DistributedTraining {

  override def checkSettings(): Unit = {
    if (settings.partitions.isDefined)
      warn("CNNs don't support partitions. This setting has no effect.")
    if (settings.batchSize.isDefined)
      warn("Setting the batch size has no effect in distributed training.")
  }

}
