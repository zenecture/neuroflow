package neuroflow.application.plugin

import java.io.{FileReader, File, PrintWriter}

import neuroflow.common.{Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{Layer, Network, WeightProvider}

import scala.io.Source
import scala.pickling.Defaults._
import scala.pickling.json._
import scala.tools.nsc.classpath.FileUtils

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

  object Json {
    def read(json: String): WeightProvider = new WeightProvider {
      def apply(v1: Seq[Layer]): Weights = JSONPickle(json).unpickle[Weights]
    }
    def write(network: Network): String = network.weights.pickle.value
  }


  object File {
    def read(path: String): WeightProvider = ~> (Source.fromFile(path).mkString) map Json.read
    def write(network: Network, path: String): Unit = ~> (new PrintWriter(new File(path))) io (_.write(Json.write(network))) io (_.close)
  }

}
