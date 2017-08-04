package neuroflow.application.processor

/**
  * @author bogdanski
  * @since 17.06.16
  */
object Normalizer {

  object MaxUnit {
    /**
      * Normalizes `xs` such that `xs.max == 1.0`
      */
    def apply(xs: Vector[Double]): Vector[Double] = xs.map(_ / xs.max)
  }



  object UnitVector {
    /**
      * Normalizes `xs` such that all vector components are <= 1.
      */
    def apply(xs: Vector[Double]): Vector[Double] = {
      val length = math.sqrt(xs.map(i => i * i).sum)
      xs.map(_ / length)
    }
  }

  object Binary {
    def apply(xs: Vector[Double]): Vector[Double] = {
      xs.map {
        case i if i > 0.0 => 1.0
        case _            => 0.0
      }
    }
  }

}
