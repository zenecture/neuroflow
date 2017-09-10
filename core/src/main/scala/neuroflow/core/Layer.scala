package neuroflow.core

/**
  * @author bogdanski
  * @since 03.01.16
  */


/** Base-label for all layers. */
trait Layer extends Serializable {
  val neurons: Int
  val symbol: String
}

trait In
trait Out
trait Hidden


/** Dense input layer carrying `neurons`. */
case class Input(neurons: Int) extends In with Layer {
  val symbol: String = "In"
}

/** Dense output layer carrying `neurons` with `activator` function. */
case class Output(neurons: Int, activator: Activator[Double]) extends Out with Layer with HasActivator[Double] {
  val symbol: String = "Out"
}

/** Dense layer carrying `neurons` with `activator` function. */
case class Dense(neurons: Int, activator: Activator[Double]) extends Layer with Hidden with HasActivator[Double] {
  val symbol: String = "Hidden"
}

/**
  * [[Focus]] is used if the desired model output
  * is not the [[Out]] layer, but a hidden one. (AutoEncoders, PCA, ...)
  */
case class Focus(inner: Layer with HasActivator[Double]) extends Layer {
  val symbol: String = s"Cluster(${inner.symbol}(${inner.activator.symbol}))"
  val neurons: Int = inner.neurons
}

case class Convolution(dimIn: (Int, Int, Int), field: Int, filters: Int,
                       stride: Int, padding: Int, activator: Activator[Double])
  extends In with Hidden with HasActivator[Double] with Layer {

  val symbol: String = "Conv"
  val dimOut: (Int, Int, Int) =
    ((dimIn._1 - field + 2 * padding) / stride + 1,
     (dimIn._2 - field + 2 * padding) / stride + 1,
      filters)
  val neurons: Int = dimOut._1 * dimOut._2 * dimOut._3 // output relevance

  private val _d1 = dimIn._1 + (2 * padding) - field
  private val _d2 = dimIn._2 + (2 * padding) - field

  assert(_d1 >= 0, s"Field $field is too big for input width ${dimIn._1 + (2 * padding)}!")
  assert(_d2 >= 0, s"Field $field is too big for input height ${dimIn._2 + (2 * padding)}!")
  assert(_d1 % stride == 0, s"Width ${_d1} doesn't match stride $stride!")
  assert(_d2 % stride == 0, s"Height ${_d2} doesn't match stride $stride!")

}
