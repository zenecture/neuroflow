package neuroflow.core

import shapeless._

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 12.06.16
  */

/**
  * Type-class witnessing that the first item within [[HList]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network starts with ${Predicate}. Check your architecture.")
trait StartsWith[L <: HList, +Predicate]

object StartsWith {

  implicit def startsWith[H, L <: HList, H0]
  (implicit eq: H =:= H0): StartsWith[H :: L, H0] = new StartsWith[H :: L, H0] {}

}

/**
  * Type-class witnessing that the last item within [[HList]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network ends with ${Predicate}. Check your architecture.")
trait EndsWith[L <: HList, +Predicate]

object EndsWith {

  implicit def hnil[P]: (P :: HNil) EndsWith P = new ((P :: HNil) EndsWith P) {}

  implicit def hlist[H, P, L <: HList]
  (implicit e: L EndsWith P): (H :: L) EndsWith P = new ((H :: L) EndsWith P) {}

}


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
