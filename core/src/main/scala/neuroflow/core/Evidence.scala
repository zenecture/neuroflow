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
