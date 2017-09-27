package neuroflow.core

import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core.Network._

import scala.concurrent.Future


/**
  * @author bogdanski
  * @since 20.01.16
  */


/**
  * Marker trait for regulators
  */
trait Regularization extends Serializable


/**
  * Continuously computes the average error for given test input and output `xs`, `ys`.
  * If the error moves too far away from the best result, measured in terms of a distance `factor`,
  * the training process will stop early to avoid over-training.
  */
case class EarlyStopping[In, Out](xs: Seq[In], ys: Seq[Out], factor: Double) extends Regularization

/**
  * The KeepBest regularization strategy takes weights, which led to the least error during training.
  * In particular, this is useful for nets whose gradients oscillate heavily during training.
  */
case object KeepBest extends Regularization


trait EarlyStoppingLogic[V] { self: Network[V, _, _] =>

  private var best = Double.PositiveInfinity
  import settings._

  def shouldStopEarly[N <: Network[_, _, _], In, Out](implicit k: CanAverage[V, N, In, Out]): Boolean = regularization match {
    case Some(es: EarlyStopping[In, Out]) =>
      val averaged = k.averagedError(es.xs, es.ys)
      if (settings.verbose) info(f"Averaged test error: $averaged%.6g. Best test error so far: $best%.6g.")
      if (averaged < best) {
        best = averaged
        false
      } else if ((averaged / best) > es.factor) {
        info(f"Early Stopping: ($averaged%.6g / $best%.6g) > ${es.factor}.")
        true
      } else false
    case _ => false
  }

}

object EarlyStoppingLogic {

  /** Type-Class for concrete net impl of error averaging. */
  trait CanAverage[V, N <: Network[_, _, _], In, Out] {
    def averagedError(xs: Seq[In], ys: Seq[Out]): Double
  }

}

trait KeepBestLogic[V] { self: Network[V, _, _] =>

  import scala.concurrent.ExecutionContext.Implicits.global

  private var bestErr = Double.PositiveInfinity
  private var bestWs: Weights[V] = self.weights

  def keepBest(error: Double): Unit = {
    if (error < bestErr) Future {
      bestErr = error
      bestWs = self.weights.map(_.copy)
    }
  }

  def takeBest(): Unit = self.settings.regularization match {
    case Some(KeepBest) =>
      info(f"Applying KeepBest strategy. Best test error so far: $bestErr%.6g.")
      weights.zip(bestWs).foreach {
        case (a, b) => a := b
      }
    case _ =>
  }

}
