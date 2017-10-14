package neuroflow.core

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 14.10.17
  */

@implicitNotFound(
  "No `CanProduce` in scope. Just import neuroflow.core._ to have it.")
trait CanProduce[A, B] {
  def apply(a: A): B
}

object CanProduce {

  implicit object DoubleCanProduceDouble extends (Double CanProduce Double) {
    def apply(double: Double) = double
  }

  implicit object DoubleCanProduceFloat extends (Double CanProduce Float) {
    def apply(double: Double) = double.toFloat
  }

}
