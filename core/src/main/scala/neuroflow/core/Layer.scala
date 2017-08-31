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


/** Fixed input layer carrying `neurons`. */
case class Input(neurons: Int) extends In with Layer {
  val symbol: String = "In"
}


/** Hidden layer carrying `neurons` with `activator` function. */
case class Hidden(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] {
  val symbol: String = "Hidden"
}


/**
  * A [[Cluster]] layer is used, where the desired model output
  * is not the [[Output]] layer, but a hidden one. (AutoEncoders, PCA, ...)
  */
case class Cluster(inner: Layer with HasActivator[Double]) extends Layer {
  val symbol: String = s"Cluster(${inner.symbol}(${inner.activator.symbol}))"
  val neurons: Int = inner.neurons
}


case class Convolution(widthIn: Int, heightIn: Int, depthIn: Int,
                      filters: Int, fieldWidth: Int, fieldHeight: Int,
                      stride: Int, padding: Int, activator: Activator[Double]) extends In with HasActivator[Double] with Layer {
  val symbol: String = "Conv"
  val widthOut: Int = (widthIn - fieldWidth + 2 * padding) / stride + 1
  val heightOut: Int = (heightIn - fieldHeight + 2 * padding) / stride + 1
  val depthOut: Int = filters
  val neurons: Int = widthOut * heightOut * depthOut
}


/** Fixed output layer carrying `neurons` with `activator` function. */
case class Output(neurons: Int, activator: Activator[Double]) extends Out with Layer with HasActivator[Double] {
  val symbol: String = "Out"
}
