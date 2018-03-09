package neuroflow.application.plugin

import java.io.{BufferedInputStream, File, FileInputStream, PrintWriter}

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import io.circe.generic.auto._
import io.circe.parser._
import io.circe.syntax._
import neuroflow.common.{CanProduce, Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{Network, WeightBreeder}
import neuroflow.dsl.Layer

import scala.collection.immutable.Stream
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
      * Deserializes weights from `json` to construct a `WeightBreeder`.
      */
    def weightBreeder[V](json: String)(implicit cp: (String CanProduce Weights[V])): WeightBreeder[V] = new WeightBreeder[V] {
      def apply(ls: Seq[Layer]): Network.Weights[V] = cp(json)
    }

    /**
      * Serializes weights of `network` to json string.
      */
    def writeWeights[V](weights: Weights[V])(implicit cp: (Weights[V] CanProduce String)): String = cp(weights)

  }


  object File {

    /**
      * Deserializes weights encoded as JSON from `file` to construct a `WeightBreeder`.
      */
    def weightBreeder[V](file: String)(implicit cp: (String CanProduce Weights[V])): WeightBreeder[V] = ~> (Source.fromFile(file).mkString) map Json.weightBreeder[V]

    /**
      * Serializes `weights` of a network to `file` using JSON.
      */
    def writeWeights[V](weights: Weights[V], file: String)(implicit cp: (Weights[V] CanProduce String)): Unit = ~> (new PrintWriter(new File(file))) io (_.write(Json.writeWeights(weights))) io (_.close)

  }

  object Jvm {

    /**
      * Gets the `File` residing at `path`.
      */
    def getResourceFile(path: String): File = new File(getClass.getClassLoader.getResource(path).toURI)

    /**
      * Gets all files within `path`.
      */
    def getResourceFiles(path: String): Seq[File] = new File(getClass.getClassLoader.getResource(path).toURI).listFiles.filter(_.isFile)

    /**
      * Gets the plain bytes from `file`.
      */
    def getBytes(file: File): Seq[Byte] = ~> (new BufferedInputStream(new FileInputStream(file))) map
      (s => (s, Stream.continually(s.read).takeWhile(_ != -1).map(_.toByte).toList)) io (_._1.close) map(_._2)

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
