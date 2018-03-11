package neuroflow.common

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 12.06.16
  */


@implicitNotFound("No `${A} CanProduce ${B}` in scope.")
trait CanProduce[A, B] {
  def apply(a: A): B
}

object CanProduce {

  // Basic type-class instances.

  implicit object DoubleCanProduceDouble extends (Double CanProduce Double) {
    def apply(double: Double): Double = double
  }

  implicit object DoubleCanProduceFloat extends (Double CanProduce Float) {
    def apply(double: Double): Float = double.toFloat
  }

  implicit object FloatCanProduceFloat extends (Float CanProduce Float) {
    def apply(float: Float): Float = float
  }

  implicit object FloatCanProduceDouble extends (Float CanProduce Double) {
    def apply(float: Float): Double = float.toDouble
  }

}

@implicitNotFound("No `TypeSize[${V}]` in scope.")
trait TypeSize[V] {
  /** Size of 1 `V` in bytes. **/
  def apply(): Int
}

object TypeSize {

  implicit object TypeSizeInt extends TypeSize[Int] {
    def apply(): Int = 4
  }

  implicit object TypeSizeFloat extends TypeSize[Float] {
    def apply(): Int = 4
  }

  implicit object TypeSizeDouble extends TypeSize[Double] {
    def apply(): Int = 8
  }

}
