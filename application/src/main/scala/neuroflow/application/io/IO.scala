package neuroflow.application.io

import java.io.FileOutputStream
import java.nio.file.{Files, Paths}

import neuroflow.common.{Logs, ByteLevel, ~>}
import neuroflow.core.Network

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO extends Logs {

  /**
    * Loads a trained network from `bytes`
    */
  def load(bytes: Array[Byte]): Option[Network] = try { Some(ByteLevel.deserialize[Network](bytes)) } catch { case ex => ~> (error(ex.toString)) next None }

  /**
    * Reads a network from file `path`.
    */
  def load(path: String): Option[Network] = try { ~> (Paths.get(path)) map Files.readAllBytes map load } catch { case ex => ~> (error(ex.toString)) next None }

  /**
    * Saves a `network` to file `path`.
    */
  def save(network: Network, path: String): Unit = ~> (new FileOutputStream(path)) io (_.write(save(network))) io (_.flush) io (_.close)

  /**
    * Saves a trained `network` to byte array.
    */
  def save(network: Network): Array[Byte] = ByteLevel.serialize(network)

}
