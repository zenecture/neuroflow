package neuroflow.application.processor

/**
  * @author bogdanski
  * @since 17.06.16
  */
object Normalizer {

  object MaxUnit {
    /**
      * Normalizes `ys` such that `ys.max == 1.0`
      */
    def apply(ys: Vector[Double]): Vector[Double] = ys.map(_ / ys.max)
  }



  object UnitVector {
    /**
      * Normalizes `ys` such that the vector components are <= 1.
      */
    def apply(ys: Vector[Double]): Vector[Double] = {
      val length = math.sqrt(ys.map(i => i * i).sum)
      ys.map(_ / length)
    }
  }

}
