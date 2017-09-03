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
  * A [[Cluster]] wrapper is used if the desired model output
  * is not the [[Out]] layer, but a hidden one. (AutoEncoders, PCA, ...)
  */
case class Cluster(inner: Layer with HasActivator[Double]) extends Layer {
  val symbol: String = s"Cluster(${inner.symbol}(${inner.activator.symbol}))"
  val neurons: Int = inner.neurons
}

case class Convolution(dimIn: (Int, Int, Int), field: (Int, Int), filters: Int,
                       stride: Int, padding: (Int, Int), activator: Activator[Double])
  extends In with Hidden with HasActivator[Double] with Layer {
  val symbol: String = "Conv"
  val dimOut: (Int, Int, Int) =
    ((dimIn._1 - field._1 + 2 * padding._1) / stride + 1,
     (dimIn._2 - field._2 + 2 * padding._2) / stride + 1,
      filters)
  val neurons: Int = dimOut._1 * dimOut._2 * dimOut._3
}
