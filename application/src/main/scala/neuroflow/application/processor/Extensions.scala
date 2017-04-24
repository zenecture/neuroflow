package neuroflow.application.processor

import breeze.linalg.DenseVector

/**
  * @author bogdanski
  * @since 17.04.17
  */
object Extensions {

  /* Lazyness */
  implicit class VectorOps(l: Vector[Double]) {
    def +(r: Vector[Double]): Vector[Double] = (l.dv + r.dv).v
    def -(r: Vector[Double]): Vector[Double] = (l.dv - r.dv).v
    def dot(r: Vector[Double]): Double = l.dv dot r.dv
    def *(r: Vector[Double]): Vector[Double] = (l.dv * r.dv).v
    def *:*(r: Vector[Double]): Vector[Double] = (l.dv *:* r.dv).v
    def /:/(r: Vector[Double]): Vector[Double] = (l.dv /:/ r.dv).v
  }

  implicit class AsDenseVector(v: Vector[Double]) {
    def dv: DenseVector[Double] = DenseVector(v.toArray)
  }

  implicit class AsVector(w: DenseVector[Double]) {
    def v: Vector[Double] = w.toArray.toVector
  }

  object cosineSimilarity {
    import breeze.linalg._
    def apply(v1: scala.Vector[Double], v2: scala.Vector[Double]): Double =
      Breeze.cosineSimilarity(DenseVector(v1.toArray), DenseVector(v2.toArray))
  }

  object Breeze {
    import breeze.generic.UFunc
    import breeze.linalg._
    import breeze.linalg.operators.OpMulInner
    object cosineSimilarity extends UFunc {
      implicit def cosineSimilarityFromDotProductAndNorm[T, U](implicit dot: OpMulInner.Impl2[T, U, Double],
                                                               normT: norm.Impl[T, Double], normU: norm.Impl[U, Double]): Impl2[T, U, Double] = {
        new Impl2[T, U, Double] {
          override def apply(v1: T, v2: U): Double = {
            val denom = norm(v1) * norm(v2)
            val dotProduct = dot(v1, v2)
            if (denom == 0.0) 0.0
            else dotProduct / denom
          }
        }
      }
    }
  }

}
