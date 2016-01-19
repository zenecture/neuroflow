package neuroflow.application.plugin

import java.io.{File, PrintWriter}

import neuroflow.common.{Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{Layer, Network, WeightProvider}

import scala.io.Source
import scala.pickling.Defaults._
import scala.pickling.json._

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

  object Json {
    /**
      * Deserializes `json` string to `WeightProvider`
      */
    def read(json: String): WeightProvider = new WeightProvider {
      def apply(v1: Seq[Layer]): Weights = JSONPickle(json).unpickle[Weights]
    }

    /**
      * Serializes `network` to json string
      */
    def write(network: Network): String = network.weights.pickle.value
  }


  object File {
    /**
      * Deserializes file from `path` to `WeightProvider`
      */
    def read(path: String): WeightProvider = ~> (Source.fromFile(path).mkString) map Json.read

    /**
      * Serializes `network` to `path` as json
      */
    def write(network: Network, path: String): Unit = ~> (new PrintWriter(new File(path))) io (_.write(Json.write(network))) io (_.close)
  }

}
