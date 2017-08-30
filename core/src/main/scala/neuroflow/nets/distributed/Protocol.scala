package neuroflow.nets.distributed

import neuroflow.core.Layer

/**
  * @author bogdanski
  * @since 28.08.17
  */

sealed trait Message extends Serializable

case object Heartbeat extends Message

case class Job(id: String, batches: Int, layers: Seq[Layer], weightDims: Seq[(Int, Int)], learningRate: Double, parallelism: Int) extends Message

case class WeightBatch(id: String, jobId: String, data: Array[(Double, Int)], position: Int) extends Message

case class ErrorBatch(id: String, jobId: String, data: Array[(Double, Int)]) extends Message

case class Ack(id: String) extends Message

case class Error(message: String, failure: Option[Throwable]) extends Message
