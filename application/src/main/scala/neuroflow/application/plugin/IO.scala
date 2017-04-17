package neuroflow.application.plugin

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import neuroflow.common.{Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{Layer, Network, WeightProvider}

import scala.io.Source

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

  object Json {

    import io.circe.parser._
    import io.circe.syntax._

    /**
      * Deserializes weights as `json` to construct a `WeightProvider`
      */
    def read(json: String): WeightProvider = new WeightProvider {
      def apply(v1: Seq[Layer]): Weights = decode[Array[Array[Double]]](json) match {
        case Left(t) => throw t
        case Right(ws) => ws.map(DenseMatrix(_))
      }
    }

    /**
      * Serializes weights of `network` to json string
      */
    def write(network: Network): String = network.weights.toArray.map(_.toArray).asJson.noSpaces
  }


  object File {
    /**
      * Deserializes weights as json from `file` to construct a `WeightProvider`
      */
    def read(file: String): WeightProvider = ~> (Source.fromFile(file).mkString) map Json.read

    /**
      * Serializes weights of `network` to `file` as json
      */
    def write(network: Network, file: String): Unit = ~> (new PrintWriter(new File(file))) io (_.write(Json.write(network))) io (_.close)
  }

}
