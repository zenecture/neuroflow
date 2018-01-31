package neuroflow.dsl

import neuroflow.core.{Softmax, SquaredMeanError}

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 03.01.16
  */

/**
  * Type-class witnessing that the first item within [[Layout]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network starts with ${Predicate}. Check your layout: ${L}")
trait StartsWith[L <: Layout, +Predicate]

object StartsWith {

  implicit def startsWith[H <: Layer, L <: Layout, H0 <: Layer]
    (implicit eq: H =:= H0): StartsWith[H :: L, H0] = new StartsWith[H :: L, H0] { }

}

/**
  * Type-class witnessing that the last item within [[Layout]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network ends with ${Predicate}. Check your layout: ${L}")
trait EndsWith[L <: Layout, +Predicate]

object EndsWith {

  implicit def hsme[P <: Layer, V]: (P :: SquaredMeanError[V]) EndsWith P = new ((P :: SquaredMeanError[V]) EndsWith P) { }

  implicit def hsmx[P <: Layer, V]: (P :: Softmax[V]) EndsWith P = new ((P :: Softmax[V]) EndsWith P) { }

  implicit def hlist[H <: Layer, P <: Layer, L <: Layout]
    (implicit e: L EndsWith P): (H :: L) EndsWith P = new ((H :: L) EndsWith P) { }

}
