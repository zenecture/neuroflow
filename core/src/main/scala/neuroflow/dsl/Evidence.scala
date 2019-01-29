package neuroflow.dsl

import neuroflow.core._

import scala.annotation.implicitNotFound

/**
  * @author bogdanski
  * @since 03.01.16
  */

@implicitNotFound("Could not prove that this layout ${L} is valid for ${N}.")
trait IsValidLayoutFor[L <: Layout, N <: Network[_, _, _]]

object IsValidLayoutFor {

  implicit def evidence_ffn[L <: Layout, N <: FFN[_]](implicit
                                                      startsWith: L StartsWith In,
                                                      endsWith: L EndsWith Out): L IsValidLayoutFor N = new IsValidLayoutFor[L, N] { }

  implicit def evidence_cnn[L <: Layout, N <: CNN[_]](implicit
                                                      startsWith: L StartsWith In,
                                                      endsWith: L EndsWith Out): L IsValidLayoutFor N = new IsValidLayoutFor[L, N] { }

  implicit def evidence_rnn[L <: Layout, N <: RNN[_]](implicit
                                                      startsWith: L StartsWith In,
                                                      endsWith: L EndsWith Out): L IsValidLayoutFor N = new IsValidLayoutFor[L, N] { }

}


@implicitNotFound("Could not prove that this network starts with ${Predicate}. Check your layout: ${L}")
trait StartsWith[L <: Layout, +Predicate]

object StartsWith {

  implicit def startsWith[H <: Layer[_], L <: Layout, H0 <: Layer[_]]
    (implicit eq: H =:= H0): StartsWith[H :: L, H0] = new StartsWith[H :: L, H0] { }

}


@implicitNotFound("Could not prove that this network ends with ${Predicate}. Check your layout: ${L}")
trait EndsWith[L <: Layout, +Predicate]

object EndsWith {

  implicit def habs[P <: Layer[_], V]: (P :: AbsCubicError[V]) EndsWith P = new ((P :: AbsCubicError[V]) EndsWith P) { }
  implicit def hsme[P <: Layer[_], V]: (P :: SquaredError[V]) EndsWith P = new ((P :: SquaredError[V]) EndsWith P) { }
  implicit def hsmx[P <: Layer[_], V]: (P :: SoftmaxLogEntropy[V]) EndsWith P = new ((P :: SoftmaxLogEntropy[V]) EndsWith P) { }
  implicit def hsmxM[P <: Layer[_], V]: (P :: SoftmaxLogMultEntropy[V]) EndsWith P = new ((P :: SoftmaxLogMultEntropy[V]) EndsWith P) { }

  implicit def hlist[H <: Layer[_], P <: Layer[_], L <: Layout]
    (implicit e: L EndsWith P): (H :: L) EndsWith P = new ((H :: L) EndsWith P) { }

}

// TODO: provide additional type classes for more rigorous proofs



