package neuroflow.application.processor

import breeze.linalg.{max, sum}
import neuroflow.application.plugin.Notation.ζ

/**
  * @author bogdanski
  * @since 17.06.16
  */
object Normalizer {

  import neuroflow.core.Network._

  object MaxUnit {
    /**
      * Normalizes `x` such that `x.max == 1.0`
      */
    def apply(x: Vector): Vector = x.map(_ / max(x))
  }

  object UnitVector {
    /**
      * Normalizes `x` such that all components are <= 1.
      */
    def apply(x: Vector): Vector = {
      val length = math.sqrt(sum(x.map(x => x * x)))
      x.map(_ / length)
    }
  }

  object Binary {
    /**
      * Turns `x` into a binary representation, where
      * components > `f` are considered to be 1, 0 otherwise.
      */
    def apply(x: Vector, f: Double = 0.0): Vector = {
      x.map {
        case i if i > f => 1.0
        case _          => 0.0
      }
    }
  }

  object ScaledVectorSpace {
    /**
      * Scales all components to be in range [-1; 1].
      */
    def apply(xs: Seq[Vector]): Seq[Vector] = {
      val max = xs.map(x => VectorLength(x)).max
      xs.map(x => x.map(_ / max))
    }
  }

  object VectorLength {
    /**
      * Computes the length of `x`.
      */
    def apply(x: Vector): Double = math.sqrt(x.map(x => x * x).sum)
  }

  object VectorFlatten {
    /**
      * Extracts the original hot vectors from horizontally merged `x`.
      */
    def apply(x: Vector): Seq[Vector] = x.data.zipWithIndex.flatMap {
      case (v, i) if v >= 1.0 => Some({ val m = ζ(x.size); m.update(i, 1.0); m })
      case (v, i) if v == 0.0 => None
      case (v, i) if v  < 0.0 => Some({ val m = ζ(x.size); m.update(i, -1.0); m })
    }.toSeq
  }

  object HotVectorIndex {
    /**
      * Locates the index of hot vector `x`.
      */
    def apply(x: Vector): Int = {
      val wi = x.data.zipWithIndex
      wi.find(_._1 == 1.0) match {
        case Some(h1) => h1._2
        case None => wi.find(_._1 == -1.0) match {
          case Some(h2) => -h2._2
          case None => throw new Exception("Doesn't work.")
        }
      }
    }
  }

  object Harmonize {
    /**
      * Harmonizes `x` by using `cap` as min/max.
      */
    def apply(x: Vector, cap: Double = 1.0): Vector = x.map {
      case v if v >  cap  =>  cap
      case v if v < -cap  => -cap
      case v              =>   v
    }
  }

}
