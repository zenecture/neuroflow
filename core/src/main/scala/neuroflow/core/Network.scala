package neuroflow.core

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg.DenseMatrix
import neuroflow.common.Logs
import neuroflow.nets.DefaultNetwork

/**
  * @author bogdanski
  * @since 03.01.16
  */

object Network {

  /**
    * Constructs a new `DefaultNetwork` with layers `ls` and settings `sets`
    */
  def apply(ls: Seq[Layer]): Network = apply(ls, Settings(false, 0.0, true))
  def apply(ls: Seq[Layer], sets: Settings): Network = new DefaultNetwork {
    val settings: Settings = sets
    val layers: Seq[Layer] = ls
    val weights: Seq[DenseMatrix[Double]] = layers.zipWithIndex.flatMap { li =>
      val (layer, index) = li
      if (index < (layers.size - 1)) {
        val (neuronsLeft, neuronsRight) = (layer.neurons, layers(index + 1).neurons)
        val product = neuronsLeft * neuronsRight
        val initialWeights = (1 to product).map(k => ThreadLocalRandom.current.nextDouble(-1, 1)).toArray
        Some(DenseMatrix.create[Double](neuronsLeft, neuronsRight, initialWeights))
      } else None
    }.toList
  }
}

/**
  * If `numericGradient` is true, the gradient will be approximated using step size `Δ`, which is alot faster
  * than actually deriving the whole net. The `verbose` flag indicates logging behavior.
  */
case class Settings(numericGradient: Boolean, Δ: Double, verbose: Boolean)

trait Network extends Logs with Serializable {
  /**
    * Settings of this neural network
    */
  val settings: Settings

  /**
    * Layers of this neural network
    */
  val layers: Seq[Layer]

  /**
    * The weights packed as a list of matrices
    */
  val weights: Seq[DenseMatrix[Double]]

  /**
    * Input `xs` and output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], stepSize: Double, precision: Double, maxIterations: Int): Unit

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): List[Double]

  override def toString: String = weights.foldLeft("")(_.toString + "\n---\n" + _.toString)
}
