package neuroflow.core

/**
  * @author bogdanski
  * @since 09.09.17
  */

/** Distributed training node */
case class Node(host: String, port: Int)


/**
  * The `messageGroupSize` controls how many weights per batch will be sent.
  * The `frameSize` is the maximum message size for inter-node communication.
  */
case class Transport(messageGroupSize: Int, frameSize: String)
