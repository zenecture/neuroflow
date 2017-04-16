package neuroflow.core

/**
  * @author bogdanski
  * @since 03.01.16
  */


/**
  * Label for any layer with `neurons` count.
  */
trait Layer {
  val neurons: Int
}

/**
  * Fixed input layer carrying `neurons`
  */
case class Input(neurons: Int) extends Layer

/**
  * Hidden layer carrying `neurons` with `activator` function
  */
case class Hidden(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double]

/**
  * A [[Cluster]] layer is used in unsupervised training scenarios, e.g. AutoEncoders,
  * where the desired model output is not the [[Output]] layer of a net, but a hidden one.
  */
case class Cluster(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double]

/**
  * Fixed output layer carrying `neurons` with `activator` function
  */
case class Output(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double]
