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
  * During training the error for `testSet` will be compared to the training error to avoid over-fitting.
  * If `stop` is set to true, the training will terminate if both errors diverge, compared by their relative `distance` factor.
  */
case class EarlyStopping(stop: Boolean, testSet: Seq[Seq[Double]], distance: Double) extends Regularization