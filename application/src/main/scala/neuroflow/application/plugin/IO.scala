package neuroflow.application.plugin

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import neuroflow.common.{Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{Layer, WeightProvider}

import scala.io.Source

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

  case class RawMatrix(rows: Int, cols: Int, precision: String, data: Array[Double]) {
    def toDenseMatrix = DenseMatrix.create[Double](rows, cols, data)
  }

  trait CanProduceRaw[V] {
    def apply(m: DenseMatrix[V]): RawMatrix
  }

  implicit object CanProduceRawFromDouble extends CanProduceRaw[Double] {
    def apply(m: DenseMatrix[Double]): RawMatrix = RawMatrix(m.rows, m.cols, "double", m.toArray)
  }

  implicit object CanProduceRawFromFloat extends CanProduceRaw[Float] {
    def apply(m: DenseMatrix[Float]): RawMatrix = RawMatrix(m.rows, m.cols, "single", m.map(_.toDouble).toArray)
  }


  object Json {

    import io.circe.generic.auto._
    import io.circe.parser._
    import io.circe.syntax._

    /**
      * Deserializes weights from `json` to construct a `WeightProvider`.
      */
    def readDouble(json: String): WeightProvider[Double] = new WeightProvider[Double] {
      def apply(v1: Seq[Layer]): Weights[Double] = decode[Array[RawMatrix]](json) match {
        case Left(t)   => throw t
        case Right(ws) => ws.map(_.toDenseMatrix)
      }
    }

    def readFloat(json: String): WeightProvider[Float] = new WeightProvider[Float] {
      def apply(v1: Seq[Layer]): Weights[Float] = decode[Array[RawMatrix]](json) match {
        case Left(t)   => throw t
        case Right(ws) => ws.map(_.toDenseMatrix.map(_.toFloat))
      }
    }

    /**
      * Serializes weights of `network` to json string.
      */
    def write[V](weights: Weights[V])(implicit c: CanProduceRaw[V]): String = weights.map(m => c(m)).asJson.noSpaces

  }


  object File {
    /**
      * Deserializes weights as json from `file` to construct a `WeightProvider`.
      */
    def readDouble(file: String): WeightProvider[Double] = ~> (Source.fromFile(file).mkString) map Json.readDouble
    def readFloat(file: String): WeightProvider[Float] = ~> (Source.fromFile(file).mkString) map Json.readFloat

    /**
      * Serializes weights of `network` to `file` as json.
      */
    def write[V](weights: Weights[V], file: String)(implicit c: CanProduceRaw[V]): Unit =
      ~> (new PrintWriter(new File(file))) io (_.write(Json.write(weights))) io (_.close)
  }

}
