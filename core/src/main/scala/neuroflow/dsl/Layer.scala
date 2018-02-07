package neuroflow.dsl

import neuroflow.core.{Activator, HasActivator}

/**
  * @author bogdanski
  * @since 03.01.16
  */


/** Base-label for all layers. */
sealed trait Layer extends Serializable {
  val neurons: Int
  val symbol: String
}

sealed trait In
sealed trait Out


/**
  * Input for a dense net, where `dimension` is
  * the number of `neurons` of this layer.
  */
case class Vector(dimension: Int) extends Layer with In {
  val symbol: String = "Vector"
  val neurons: Int = dimension
}

/**
  * A dense layer is fully connected, with `neurons` in this layer.
  * The `activator` function gets applied on the output element wise.
  */
case class Dense[V](neurons: Int, activator: Activator[V]) extends Layer with Out with HasActivator[V] {
  val symbol: String = "Dense"
}

/**
  * A focus layer is used if the desired model output is not
  * the [[Out]] layer, but the `inner` one. (AutoEncoders, PCA, ...)
  */
case class Focus[V](inner: Layer with HasActivator[V]) extends Layer {
  val symbol: String = s"Focus(${inner.symbol}(${inner.activator.symbol}))"
  val neurons: Int = inner.neurons
}

object Ω { // Alias syntax for Focus
  def apply[V](inner: Layer with HasActivator[V]): Focus[V] = Focus(inner)
}

/**
  *
  * Convolutes the input [[neuroflow.common.Tensor3D]], where:
  *
  *   `dimIn`      Input dimension. (x, y, z)
  *   `padding`    A padding can be specified to ensure full convolution. (x, y)
  *   `field`      The receptive field. (x, y)
  *   `filters`    Number of filters attached to the input.
  *   `stride`     Sliding the receptive field over the input volume with stride. (x, y)
  *   `activator`  The activator function gets applied on the output element-wise.
  *
  */
case class Convolution[V](dimIn      :  (Int, Int, Int),
                          padding    :  (Int, Int),
                          field      :  (Int, Int),
                          stride     :  (Int, Int),
                          filters    :   Int,
                          activator  :   Activator[V]) extends Layer with HasActivator[V] with In {

  val symbol: String = "Convolution"

  val dimInPadded: (Int, Int, Int) =
    (dimIn._1 + (2 * padding._1),
     dimIn._2 + (2 * padding._2),
     dimIn._3)

  val dimOut: (Int, Int, Int) =
    ((dimIn._1 + (2 * padding._1) - field._1) / stride._1 + 1,
     (dimIn._2 + (2 * padding._2) - field._2) / stride._2 + 1,
      filters)

  val neurons: Int = dimOut._1 * dimOut._2 * dimOut._3 // output relevance

  private val _d1 = dimIn._1 + (2 * padding._1) - field._1
  private val _d2 = dimIn._2 + (2 * padding._2) - field._2

  assert(filters > 0, "Filters must be positive!")
  assert(stride._1 > 0 && stride._2 > 0, "Strides must be positive!")
  assert(field._1 > 0 && field._2 > 0, "Field must be positive!")
  assert(dimIn._1 > 0 && dimIn._2 > 0 && dimIn._3 > 0, "Input dimension must be positive!")
  assert(_d1 >= 0, s"Field $field is too big for input width ${dimIn._1}!")
  assert(_d2 >= 0, s"Field $field is too big for input height ${dimIn._2}!")
  assert(_d1 % stride._1 == 0, s"W + 2P - F % S != 0: Width W = ${dimIn._1} doesn't work with stride S = ${stride._1}, field F = ${field._1}, padding P = ${padding._1}!")
  assert(_d2 % stride._2 == 0, s"H + 2P - F % S != 0: Height H = ${dimIn._2} doesn't work with stride S = ${stride._2}, field F = ${field._2}, padding P = ${padding._2}!")

}

object Convolution {

  implicit class IntTupler(i: Int) {
    def `²`: (Int, Int) = (i, i)
    def `³`: (Int, Int, Int) = (i, i, i)
  }

}
