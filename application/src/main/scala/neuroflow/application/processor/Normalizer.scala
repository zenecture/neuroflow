package neuroflow.application.processor

import neuroflow.application.plugin.Notation.ζ

/**
  * @author bogdanski
  * @since 17.06.16
  */
object Normalizer {

  object MaxUnit {
    /**
      * Normalizes `x` such that `x.max == 1.0`
      */
    def apply(x: Vector[Double]): Vector[Double] = x.map(_ / x.max)
  }

  object UnitVector {
    /**
      * Normalizes `x` such that all components are <= 1.
      */
    def apply(x: Vector[Double]): Vector[Double] = {
      val length = math.sqrt(x.map(x => x * x).sum)
      x.map(_ / length)
    }
  }

  object Binary {
    /**
      * Turns `x` into a binary representation, where
      * components > `f` are considered to be 1, 0 otherwise.
      */
    def apply(x: Vector[Double], f: Double = 0.0): Vector[Double] = {
      x.map {
        case i if i > f => 1.0
        case _          => 0.0
      }
    }
  }

  object VectorLength {
    /**
      * Computes the length of `x`.
      */
    def apply(x: Vector[Double]): Double = math.sqrt(x.map(x => x * x).sum)
  }

  object VectorFlatten {
    /**
      * Extracts the original hot vectors from horizontally merged `x`.
      */
    def apply(x: Vector[Double]): Seq[Vector[Double]] = x.zipWithIndex.flatMap {
      case (v, i) if v >= 1.0 => Some(ζ(x.size).updated(i, 1.0))
      case (v, i)             => None
    }
  }

  object HotVectorIndex {
    /**
      * Locates the index of hot vector `x`.
      */
    def apply(x: Vector[Double]): Int = {
      if (x.count(_ == 1.0) > 1) throw new Exception("Doesn't work.")
      else x.zipWithIndex.find(_._1 == 1.0) match {
        case Some((v, i)) => i
        case None         => throw new Exception("Doesn't work.")
      }
    }
  }

}
