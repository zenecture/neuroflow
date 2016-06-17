package neuroflow.application.processor

/**
  * @author bogdanski
  * @since 17.06.16
  */
object Normalizer {


  /**
    * Normalizes `ys` such that `ys.max == 1.0`
    */
  def apply(ys: Seq[Double]) = ys.map(_ / ys.max)

}
