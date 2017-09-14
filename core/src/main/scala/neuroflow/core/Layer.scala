package neuroflow.core

/**
  * @author bogdanski
  * @since 03.01.16
  */


/** Base-label for all layers. */
sealed trait Layer extends Serializable {
  val neurons: Int
  val symbol: String
}

sealed trait In
sealed trait Out
sealed trait Hidden


/** Dense input layer carrying `neurons`. */
case class Input(neurons: Int) extends Layer with In {
  val symbol: String = "In"
}

/** Dense output layer carrying `neurons` with `activator` function. */
case class Output(neurons: Int, activator: Activator[Double]) extends Layer with HasActivator[Double] with Out {
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

/**
  * Convolutes the input volume. Where:
  *   Input dimension `in` as (width, height, depth).
  *   The receptive `field` as (width, height).
  *   How many independent `filters` are attached to the input.
  *   Sliding over the input volume using a `stride`.
  *   Adds a `padding` to the input volume, width and height.
  *   Applying the `activator` function element-wise.
  */
case class Convolution(in: (Int, Int, Int), field: (Int, Int), filters: Int,
                       stride: Int, padding: Int, activator: Activator[Double])
  extends Hidden with HasActivator[Double] with Layer with In {

  val symbol: String = "Conv"

  val dimIn: (Int, Int, Int) =
     (in._1 + (2 * padding),
      in._2 + (2 * padding),
      in._3)

  val dimOut: (Int, Int, Int) =
    ((dimIn._1 - field._1) / stride + 1,
     (dimIn._2 - field._2) / stride + 1,
      filters)

  val neurons: Int = dimOut._1 * dimOut._2 * dimOut._3 // output relevance

  private val _d1 = in._1 + (2 * padding) - field._1
  private val _d2 = in._2 + (2 * padding) - field._2

  assert(filters  > 0, "Filters must be positive!")
  assert(stride   > 0, "Stride must be positive!")
  assert(padding >= 0, "Padding must be non-negative!")
  assert(field._1 > 0 && field._2 > 0, "Field must be positive!")
  assert(in._1 > 0 && in._2 > 0 && in._3 > 0, "Input dimension must be positive!")
  assert(_d1 >= 0, s"Field $field is too big for input width ${in._1 + (2 * padding)}!")
  assert(_d2 >= 0, s"Field $field is too big for input height ${in._2 + (2 * padding)}!")
  assert(_d1 % stride == 0, s"Width ${_d1} doesn't match stride $stride!")
  assert(_d2 % stride == 0, s"Height ${_d2} doesn't match stride $stride!")

}

object Convolution {

  implicit class IntTupler(i: Int) {
    def `²`: (Int, Int) = (i, i)
    def `³`: (Int, Int, Int) = (i, i, i)
  }

}
