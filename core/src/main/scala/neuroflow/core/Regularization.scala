package neuroflow.core


/**
  * @author bogdanski
  * @since 20.01.16
  */


/**
  * Marker trait for regulators
  */
trait Regularization extends Serializable


/**
  * Use training set `xs` with desired output `ys` to stop the training process
  * if the error on this training set begins to increase due to over-training.
  * The distance `factor` triggers the drop out.
  */
case class EarlyStopping(xs: Seq[Seq[Double]], ys: Seq[Seq[Double]], factor: Double) extends Regularization

trait EarlyStoppingLogic { self: Network =>

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
