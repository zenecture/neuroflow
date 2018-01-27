package neuroflow.application.plugin

import breeze.generic.UFunc
import breeze.linalg.{DenseVector, _}
import breeze.linalg.operators.OpMulInner

import scala.reflect.ClassTag


/**
  * @author bogdanski
  * @since 17.04.17
  */
object Extensions {

  /* Enriched scala.Vector */
  implicit class VectorOps(l: scala.Vector[Double]) {
    def +(r: scala.Vector[Double]): scala.Vector[Double]   = (l.dv  +  r.dv).vv
    def -(r: scala.Vector[Double]): scala.Vector[Double]   = (l.dv  -  r.dv).vv
    def dot(r: scala.Vector[Double]): Double               =  l.dv dot r.dv
    def *(r: scala.Vector[Double]): scala.Vector[Double]   = (l.dv  *  r.dv).vv
    def *:*(r: scala.Vector[Double]): scala.Vector[Double] = (l.dv *:* r.dv).vv
    def /:/(r: scala.Vector[Double]): scala.Vector[Double] = (l.dv /:/ r.dv).vv
  }

  object scalaVectorCosineSimilarity {
    def apply(v1: scala.Vector[Double], v2: scala.Vector[Double]): Double =
      Extensions.cosineSimilarity(DenseVector(v1.toArray), DenseVector(v2.toArray))
  }

  object scalaVectorEuclideanDistance {
    def apply(v1: scala.Vector[Double], v2: scala.Vector[Double]): Double =
      breeze.linalg.functions.euclideanDistance(v1.dv, v2.dv)
  }

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

  implicit class AsDenseVector[V: ClassTag](v: scala.Vector[V]) {
    def dv: DenseVector[V] = DenseVector(v.toArray)
  }

  implicit class AsVector[V: ClassTag](w: DenseVector[V]) {
    def vv: scala.Vector[V] = w.toArray.toVector
  }

  implicit class DenseVectorDoubleToFloat(v: DenseVector[Double]) {
    def float: DenseVector[Float] = v.map(_.toFloat)
  }

  implicit class DenseVectorFloatToDouble(v: DenseVector[Float]) {
    def double: DenseVector[Double] = v.map(_.toDouble)
  }

}
