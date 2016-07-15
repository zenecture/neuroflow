package neuroflow.core

/**
  * @author bogdanski
  * @since 14.07.16
  */

object ∞ {

  import Network.Vector

  /**
    * Generates a positive infinity [[Network.Vector]] for specified `dimension`.
    */
  def apply(dimension: Int): Vector = (0 until dimension) map (_ => Double.PositiveInfinity) toVector

}
