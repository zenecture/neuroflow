package neuroflow.application.io

import java.nio.file.{Files, Paths}

import neuroflow.common.{Logs, ~>}
import neuroflow.core.Network.Weights
import neuroflow.core.{Layer, WeightProvider}

import scala.pickling._
import scala.pickling.Defaults.genOpenSumUnpickler
import scala.pickling.binary._

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

//  /**
//    * Loads a trained network from `bytes`
//    */
//  def load(bytes: Array[Byte]): Option[Network] = try { Some(ByteLevel.deserialize[Network](bytes)) } catch { case ex => ~> (error(ex.toString)) next None }

//  /**
//    * Reads a network from file `path`.
//    */
//  def load(path: String): Option[Weights] = try { ~> (Paths.get(path)) map Files.readAllBytes map load } catch { case ex => ~> (error(ex.toString)) next None }
//
//  /**
//    * Saves a `network` to file `path`.
//    */
//  def save(network: Network, path: String): Unit = ~> (new FileOutputStream(path)) io (_.write(save(network))) io (_.flush) io (_.close)
//
////  /**
////    * Saves a trained `network` to byte array.
////    */
////  def save(network: Network): Array[Byte] = ByteLevel.serialize(network)
//
//  def save(network: Network): Array[Byte] = ??? /*{
//    implicit val pickler = Pickler.generate[Weights]
//    val pickled = network.weights.pickle
//    pickled.value
//  }*/
//
//  def load(bytes: Array[Byte]): Option[Weights] =  ??? /*{
//    implicit val unpickler = Unpickler.generate[Weights]
//    bytes.unpickle[Weights]
//  }*/
//
//
//  def huarz = {
//
//  }

//  def unapply(bytes: Array[Byte]): WeightProvider = {
//    //implicit val unpickler = Unpickler.generate[Weights]
//    new WeightProvider {
//      def apply(v1: Seq[Layer]): Weights = bytes.unpickle[Weights]
//    }
//  }
//
//  def unapply(path: String): WeightProvider = ~> (Paths.get(path)) map Files.readAllBytes map unapply

}
