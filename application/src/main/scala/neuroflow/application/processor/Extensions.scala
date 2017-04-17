package neuroflow.application.processor

import breeze.generic.UFunc
import breeze.linalg._
import breeze.linalg.operators.OpMulInner

/**
  * @author bogdanski
  * @since 17.04.17
  */
object Extensions {

  implicit class SeqLikeVector(l: Seq[Double]) {
    def +(r: Seq[Double]): Seq[Double] = (l zip r).map(l => l._1 + l._2)
  }

  object cosineSimilarity {
    def apply(v: Seq[Double], v2: Seq[Double]): Double =
      Breeze.cosineSimilarity(DenseVector(v.toArray), DenseVector(v2.toArray))
  }

  object Breeze {
    object cosineSimilarity extends UFunc {
      implicit def cosineSimilarityFromDotProductAndNorm[T, U](implicit dot: OpMulInner.Impl2[T, U, Double],
                                                               normT: norm.Impl[T, Double], normU: norm.Impl[U, Double]): Impl2[T, U, Double] = {
        new Impl2[T, U, Double] {
          override def apply(v: T, v2: U): Double = {
            val denom = norm(v) * norm(v2)
            val dotProduct = dot(v, v2)
            if (denom == 0.0) 0.0
            else dotProduct / denom
          }
        }
      }
    }
  }

}
