package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}

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
  val filters: Int
  val fieldSize: Int
  val stride: Int
  val padding: Int
  val reshape: Option[Int]
  def receptiveField(in: Matrix): Matrices
}

/** Convolutes the input in a linear fashion. */
case class LinConvolution(filters: Int,
                          fieldSize: Int,
                          stride: Int,
                          padding: Int,
                          activator: Activator[Double],
                          reshape: Option[Int] = None) extends Layer with Convolutable {
  import Network._
  val neurons: Int = reshape.getOrElse(filters * fieldSize)
  val symbol: String = "Cn"
  private val pads  = DenseVector.zeros[Double](padding)
  def receptiveField(in: Matrix): Matrices = {
    (0 until in.rows).map { r =>
      val d = DenseVector.vertcat(pads, in.t(::, r), pads).toArray
      (1 to filters).toParArray.flatMap { _ =>
        Range(0, d.size - fieldSize, stride).toParArray.map { i =>
          DenseMatrix.create[Double](fieldSize, 1, d.slice(i, i + fieldSize))
        }
      }.reduce((l, r) => DenseMatrix.horzcat(l, r))
    }
  }
}


/** Fixed output layer carrying `neurons` with `activator` function */
case class Output(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] {
  val symbol: String = "Out"
}
