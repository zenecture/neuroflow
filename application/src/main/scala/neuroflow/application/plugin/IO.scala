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
      * Deserializes weights as `json` to construct a `WeightProvider`
      */
    def read(json: String): WeightProvider = new WeightProvider {
      def apply(v1: Seq[Layer]): Weights = JSONPickle(json).unpickle[Weights]
    }

    /**
      * Serializes weights of `network` to json string
      */
    def write(network: Network): String = network.weights.pickle.value
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
