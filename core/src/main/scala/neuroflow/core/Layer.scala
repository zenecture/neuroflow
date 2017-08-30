package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author bogdanski
  * @since 03.01.16
  */


/** Base-label for all layers. */
trait Layer extends Serializable {
  val neurons: Int
  val symbol: String
}


/** Fixed input layer carrying `neurons` */
case class Input(neurons: Int) extends Layer {
  val symbol: String = "In"
}


/** Hidden layer carrying `neurons` with `activator` function */
case class Hidden(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] {
  val symbol: String = "Hidden"
}


/**
  * A [[Cluster]] layer is used in (un-)supervised training scenarios, e.g. AutoEncoders,
  * where the desired model output is not the [[Output]] layer of a net, but a hidden one.
  */
case class Cluster(inner: Layer with HasActivator[Double]) extends Layer {
  val symbol: String = s"Cluster(${inner.symbol}(${inner.activator.symbol}))"
  val neurons: Int = inner.neurons
}


/**
  * Convolutes the input using implementation `receptiveField`,
  * which transforms the input to a matrix holding all field slices for all filters.
  * The `activator` function will be mapped over the resulting filters.
  * `filters`: amount of filters for this layer
  * `fieldSize`: the size of the field
  * `stride`: the stride to use iterating over the input
  * `padding`: adds zero-padding to the input
  * `reshape`: reshapes the output to a matrix of shape [1, reshape], so fully layers can dock
  */
trait Convolutable extends HasActivator[Double] {
  import Network._
  val width: Int
  val height: Int
  val depth: Int
  val filters: Int
  val fieldSize: Int
  val stride: Int
  val padding: Int
  val reshape: Option[Int]
  def receptiveField(in: Matrices): Matrices
}


/** Fixed output layer carrying `neurons` with `activator` function */
case class Output(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] {
  val symbol: String = "Out"
}
