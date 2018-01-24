package neuroflow.application.plugin

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import io.circe.generic.auto._
import io.circe.parser._
import io.circe.syntax._
import neuroflow.common.{Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{CanProduce, Layer, Network, WeightProvider}

import scala.io.Source

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

  case class RawMatrix[V: Zero](rows: Int, cols: Int, precision: String, data: Array[V]) {
    def toDenseMatrix: DenseMatrix[V] = DenseMatrix.create[V](rows, cols, data)
  }

  object Json {

    /**
      * Deserializes weights from `json` to construct a `WeightProvider`.
      */
    def read[V](json: String)(implicit cp: (String CanProduce Weights[V])): WeightProvider[V] = new WeightProvider[V] {
      def apply(ls: Seq[Layer]): Network.Weights[V] = cp(json)
    }

    /**
      * Serializes weights of `network` to json string.
      */
    def write[V](weights: Weights[V])(implicit cp: (Weights[V] CanProduce String)): String = cp(weights)

  }


  object File {

    /**
      * Deserializes weights as json from `file` to construct a `WeightProvider`.
      */
    def read[V](file: String)(implicit cp: (String CanProduce Weights[V])): WeightProvider[V] = ~> (Source.fromFile(file).mkString) map Json.read[V]

    /**
      * Serializes weights of `network` to `file` as json.
      */
    def write[V](weights: Weights[V], file: String)(implicit cp: (Weights[V] CanProduce String)): Unit = ~> (new PrintWriter(new File(file))) io (_.write(Json.write(weights))) io (_.close)

  }

  /**
    * Type-Classes
    */

  implicit object DoubleWeightsToJson extends (Weights[Double] CanProduce String) {
    def apply(ws: Weights[Double]): String = ws.map(m => RawMatrix(m.rows, m.cols, "double", m.toArray)).toArray.asJson.noSpaces
  }

  implicit object FloatWeightsToJson extends (Weights[Float] CanProduce String) {
    def apply(ws: Weights[Float]): String = ws.map(m => RawMatrix(m.rows, m.cols, "single", m.toArray)).toArray.asJson.noSpaces
  }

  implicit object JsonToDoubleWeights extends (String CanProduce Weights[Double]) {
    def apply(json: String): Weights[Double] = decode[Array[RawMatrix[Double]]](json) match {
      case Left(t)   => throw t
      case Right(ws) => ws.map(_.toDenseMatrix)
    }
  }

  implicit object JsonToFloatWeights extends (String CanProduce Weights[Float]) {
    def apply(json: String): Weights[Float] = decode[Array[RawMatrix[Float]]](json) match {
      case Left(t)   => throw t
      case Right(ws) => ws.map(_.toDenseMatrix)
    }
  }

}
