package neuroflow.core

import breeze.linalg.operators.{OpDiv, OpSub}
import breeze.math._
import neuroflow.core.Network._

/**
  * @author bogdanski
  * @since 21.01.16
  */


/**
  * Approximates `weight` residing in respective `layer` in `weights` using `lossFunction`.
  * For GPU implementations calling `sync` between updates is necessary.
  */
trait Approximation[V] extends Serializable {
  def apply(weights: Weights[V], lossFunction: () => Matrix[V], sync: () => Unit, layer: Int, weight: (Int, Int))
           (implicit
            ring: Semiring[V],
            div: OpDiv.Impl2[Matrix[V], V, Matrix[V]],
            sub: OpSub.Impl2[Matrix[V], Matrix[V], Matrix[V]]): Matrix[V]
}

/**
  * Approximates using centralized finite diffs with step size `Δ`.
  */
case class FiniteDifferences[V: Numeric](Δ: V) extends Approximation[V] {

  import Numeric.Implicits._

  def apply(weights: Weights[V], lossFunction: () => Matrix[V], sync: () => Unit, layer: Int, weight: (Int, Int))
           (implicit
            ring: Semiring[V],
            div: OpDiv.Impl2[Matrix[V], V, Matrix[V]],
            sub: OpSub.Impl2[Matrix[V], Matrix[V], Matrix[V]]): Matrix[V] = {

    val w = weights(layer)(weight)
    weights(layer).update(weight, w - Δ)
    sync()

    val a = lossFunction()
    weights(layer).update(weight, w + Δ)
    sync()

    val b = lossFunction()
    weights(layer).update(weight, w)
    sync()

    val `2` = ring.+(ring.one, ring.one)
    val delta = ring.*(`2`, Δ)

    div(b - a, delta)

  }

}
