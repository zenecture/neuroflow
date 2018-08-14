package neuroflow.dsl

import breeze.linalg.DenseVector
import neuroflow.core.{Activator, HasActivator, Tensor3D}

/**
  * @author bogdanski
  * @since 03.01.16
  */


sealed trait Layer extends Serializable {

  /**
    * Algebraic representation
    */
  type algebraicType

  /**
    * Overall output relevance
    */
  val neurons: Int

  /**
    * Unique layer symbol
    */
  val symbol: String

}



sealed trait In
sealed trait Out



/**
  * Input vector for a dense net, with `neurons` in this layer.
  * Optionally, an `activator` can be applied element wise.
  */
case class Vector[V](dimension: Int, activator: Option[Activator[V]] = None) extends Layer with In {
  type algebraicType = DenseVector[V]
  val symbol: String = "Vector"
  val neurons: Int = dimension
}



/**
  * A dense layer is fully connected, with `neurons` in this layer.
  * The `activator` function gets applied on the output element wise.
  */
case class Dense[V](neurons: Int, activator: Activator[V]) extends Layer with Out with HasActivator[V] {
  type algebraicType = DenseVector[V]
  val symbol: String = "Dense"
}



/**
  *
  * Convolutes the input [[Tensor3D]], producing a new one, where:
  *
  *   `dimIn`      Input tensor dimension. (x, y, z)
  *   `dimOut`     Output tensor dimension. (x, y, z)
  *   `padding`    A padding can be specified to ensure full convolution. (x, y)
  *   `field`      The size of the receptive field. (x, y)
  *   `filters`    Number of filters attached to the input. (dimOut.z = filters)
  *   `stride`     Striding the receptive field over the input tensor. (x, y)
  *   `activator`  The activator function gets applied on the output tensor.
  *
  */
case class Convolution[V](dimIn      :  (Int, Int, Int),
                          padding    :  (Int, Int),
                          field      :  (Int, Int),
                          stride     :  (Int, Int),
                          filters    :   Int,
                          activator  :   Activator[V]) extends Layer with HasActivator[V] with In {

  type algebraicType = Tensor3D[V]

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

  /**
    * Import these for short syntax [[Convolution]] parameterization.
    */

  implicit def autoTupler(i: Int): (Int, Int) = (i, i)
  implicit def autoTripler(i: Int): (Int, Int, Int) = (i, i, i)

}


