package neuroflow.nets

import breeze.linalg.DenseMatrix
import neuroflow.core.{Settings, Layer, Network}

/**
  *
  * A shared network will constrain certain weights to have the same value.
  * This will lead to better performance for convoluted net architectures,
  * as well as more stable network with respectto translations and distortions
  * of an (still invariant) input.
  *
  * This is TODO.
  *
  * @author bogdanski
  * @since 15.01.16
  */
trait SharedNetwork extends Network {
  val settings: Settings = _
  val weights: List[DenseMatrix[Double]] = _
  val layers: Seq[Layer] = _

  /**
    * Input `xs` will be evaluated based on current weights
    */
  def evaluate(xs: Seq[Double]): List[Double] = ???

  /**
    * Input `xs` and output `ys` will be the mold for the weights.
    * Returns this `Network`, with new weights.
    */
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]]): Unit = ???
  def train(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], stepSize: Double, precision: Double, maxIterations: Int): Unit = ???

}
