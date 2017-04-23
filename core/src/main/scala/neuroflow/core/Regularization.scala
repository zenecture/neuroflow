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
  * Continuously computes the average error for given test input and output vectors `xs`, `ys`.
  * If the error moves too far away from the best result, measured in terms of a distance `factor`,
  * the training process will stop early to avoid over-training.
  */
case class EarlyStopping(xs: Seq[Vector], ys: Seq[Vector], factor: Double) extends Regularization

/**
  * The KeepBest regularization strategy takes weights, which led to the least error during training.
  * In particular, this is useful for RNNs, as the error can oscillate heavily during training.
  */
case object KeepBest extends Regularization


trait EarlyStoppingLogic { self: Network =>

  private var best = Double.PositiveInfinity
  import settings._

  def shouldStopEarly[N <: Network](net: N)(implicit k: CanAverage[N]): Boolean = regularization match {
    case Some(EarlyStopping(xs, ys, f)) =>
      val averaged = k.averagedError(xs, ys)
      if (settings.verbose) info(f"Averaged test error: $averaged%.6g. Best test error so far: $best%.6g.")
      if (averaged < best) {
        best = averaged
        false
      } else if ((averaged / best) > f) {
        info(f"Early Stopping: ($averaged%.6g / $best%.6g) > $f.")
        true
      } else false
    case _ => false
  }

}

object EarlyStoppingLogic {

  /** Type-Class for concrete net impl of error averaging. */
  trait CanAverage[N <: Network] {
    def averagedError(xs: Seq[Vector], ys: Seq[Vector]): Double
  }

}

trait KeepBestLogic { self: Network =>

  import scala.concurrent.ExecutionContext.Implicits.global

  private var bestErr = Double.PositiveInfinity
  private var bestWs: Weights = self.weights

  def update(error: Double, ws: Weights): Unit = {
    if (error < bestErr) Future {
      bestErr = error
      bestWs = ws.map(_.copy)
    }
  }

  def take(): Unit = self.settings.regularization match {
    case Some(KeepBest) =>
      info(f"Applying KeepBest strategy. Best test error so far: $bestErr%.6g.")
      bestWs.foreach { l =>
        l.foreachPair { (k, v) =>
          self.weights(bestWs.indexOf(l)).update(k, v)
        }
      }
    case _ =>
  }

}
