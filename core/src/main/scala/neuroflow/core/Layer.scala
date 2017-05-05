package neuroflow.core

import breeze.linalg.DenseMatrix

/**
  * @author bogdanski
  * @since 03.01.16
  */


/** Base-label for all layers. */
trait Layer {
  val neurons: Int
  val symbol: String
}


/** Fixed input layer carrying `neurons` */
case class Input(neurons: Int) extends Layer {
  val symbol: String = "In"
}


/** Hidden layer carrying `neurons` with `activator` function */
case class Hidden(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] {
  val symbol: String = "H"
}


/**
  * A [[Cluster]] layer is used in (un-)supervised training scenarios, e.g. AutoEncoders,
  * where the desired model output is not the [[Output]] layer of a net, but a hidden one.
  */
case class Cluster(inner: Layer with HasActivator[Double]) extends Layer {
  val symbol: String = s"Cl(${inner.symbol}(${inner.activator.symbol}))"
  val neurons: Int = inner.neurons
}


/**
  * Convolutes the input using `receptiveField`,
  * which transforms the input to a matrix holding all field slices.
  * `filters`: amount of filters for this layer
  * `fieldSize`: the size of the field
  * `stride`: the stride, or stepsize
  * `padding`: add zero-padding to input
  */
trait Convolutable extends HasActivator[Double] {
  import Network._
  val filters: Int
  val fieldSize: Int
  val stride: Int
  val padding: Int
  def receptiveField(in: Matrix): Matrix
}

/** Convolutes the input in a linear fashion. */
case class LinConvolution(filters: Int,
                          fieldSize: Int,
                          stride: Int,
                          padding: Int,
                          activator: Activator[Double]) extends Layer with Convolutable {
  import Network._
  val neurons: Int = filters * filters * fieldSize
  val symbol: String = "Cn"
  def receptiveField(in: Matrix): Matrix = {
    val pads  = DenseMatrix.zeros[Double](1, padding)
    val input = DenseMatrix.horzcat(pads, in, pads)
    (1 to filters).toParArray.flatMap { _ =>
      Range(0, input.size - fieldSize, stride).toParArray.map { i =>
        DenseMatrix.create[Double](fieldSize, 1, input.data.slice(i, i + fieldSize))
      }
    }.reduce((l, r) => DenseMatrix.horzcat(l, r))
  }
}


/** Fixed output layer carrying `neurons` with `activator` function */
case class Output(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] {
  val symbol: String = "Out"
}
