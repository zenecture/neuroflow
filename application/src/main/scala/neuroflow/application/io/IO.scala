package neuroflow.application.io

import neuroflow.common.ByteLevel
import neuroflow.core.Network

/**
  * @author bogdanski
  * @since 09.01.16
  */
object IO {

  /**
    * Loads a trained network from `bytes`
    */
  def load(bytes: Array[Byte]): Option[Network] = try {
    Some(ByteLevel.deserialize[Network](bytes))
  } catch {
    case ex => None
  }


  /**
    * Reads a network from file `path`.
    */
  def load(path: String): Option[Network] = ???

  /**
    * Saves a `network` to file `path`.
    */
  def save(network: Network, path: String): Unit = ???

  /**
    * Saves a trained `network` to byte array.
    */
  def save(network: Network): Array[Byte] = ByteLevel.serialize(network)

}
