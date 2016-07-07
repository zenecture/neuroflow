package neuroflow.core

import neuroflow.core.Network.Vector


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


// TODO: Make this trait generic for all nets and use type classes for concrete impls.
trait EarlyStoppingLogic { self: FeedForwardNetwork =>

  var best = Double.PositiveInfinity
  import settings._

  def shouldStopEarly: Boolean = {
    if (regularization.isEmpty) false
    else {
      val r = regularization.map { case es: EarlyStopping => es }.get
      val errors = r.xs.map(evaluate).zip(r.ys).map {
        case (a, b) =>
          val im = a.zip(b).map {
            case (x, y) => (x - y).abs
          }
          im.sum / im.size.toDouble
      }
      val averaged = errors.sum / errors.size.toDouble
      if (averaged < best) {
        best = averaged; false
      } else if ((averaged / best) > r.factor) {
        if (settings.verbose) info(s"Early Stopping: ($averaged / $best) > ${r.factor}.")
        true
      } else false
    }
  }

}
