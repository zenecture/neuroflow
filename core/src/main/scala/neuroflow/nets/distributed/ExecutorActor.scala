package neuroflow.nets.distributed

import java.util.UUID

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import breeze.linalg.DenseMatrix
import neuroflow.common.Logs
import neuroflow.core._
import neuroflow.core.Network._

import scala.collection.IndexedSeq
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._

/**
  * @author bogdanski
  * @since 11.09.17
  */

abstract class ExecutorActor[In <: Seq[_], Out <: Seq[_]](xs: In, ys: Out, settings: Settings[Double]) extends Actor with Logs {

  import context.dispatcher

  private val _scheduler = context.system.scheduler
  private var _resender = _scheduler.schedule(1 minute, 10 seconds, self, 'Resend)

  private val _MSGGS = settings.transport.messageGroupSize

  private val _weights = ArrayBuffer.empty[DenseMatrix[Double]]
  private var _job: Job = _
  private var _batchCount = 0
  private val _buffer = ArrayBuffer.empty[WeightBatch]

  private val _acks = collection.mutable.HashMap.empty[String, Message]
  private var _request: ActorRef = _

  def receive = {

    case Heartbeat =>
      info(s"Connected to $sender.")
      sender ! Heartbeat

    case job@Job(id, _, _, weightDims, _, _) =>
      resetState()
      _job = job
      _request = sender
      weightDims.foreach {
        case (rows, cols) =>
          _weights += DenseMatrix.create[Double](rows, cols, Array.fill(rows * cols)(0.0))
      }

    case 'Resend =>
      debug(s"Resending ${_acks.size} batches ...")
      _acks.values.foreach(_request ! _)

    case Ack(id) =>
      debug(s"Got ack $id")
      _acks -= id

    case batch@WeightBatch(id, jobId, _, _) if _job == null =>
      debug(s"Got batch, but no job for id = $jobId. Buffering ...")
      sender ! Ack(id)
      _buffer += batch
      _scheduler.scheduleOnce(3 seconds, self, 'ReprocessBuffer)

    case b@WeightBatch(id, jobId, _, position) if position < 0 =>
      debug(s"Got malformed WeightBatch $b. Discarding it ...")

    case WeightBatch(id, jobId, _, _) if jobId != _job.id =>
      debug(s"Got batch for old job with id = $jobId. Discarding it ...")

    case WeightBatch(id, jobId, data, position) if jobId == _job.id =>
      debug(s"Got WeightBatch for jobId = $jobId")
      sender ! Ack(id)
      _batchCount += 1
      data.foreach { case (v, i) => _weights(position).data.update(i, v) }
      val isComplete = _batchCount >= _job.batches
      if (isComplete) {
        info(s"Executing job with ${xs.size} samples and ${_weights.map(_.size).sum} weights ...")
        val (weights, error) = compute(xs, ys, _job.layers, _weights, _job.learningRate, _job.parallelism)
        info("... done. Sending back ...")
        sendResults(_job, weights, error, _request)
        resetState()
      }

    case 'ReprocessBuffer =>
      _buffer.foreach(self ! _)
      _buffer.clear()

  }

  private def resetState(): Unit = {
    _job = null
    _batchCount = 0
    _weights.clear()
  }

  private def sendResults(job: Job, weights: Weights[Double], error: DenseMatrix[Double], sender: ActorRef): Unit = {
    val weightsWi = weights.map(_.data.zipWithIndex.grouped(_MSGGS)).zipWithIndex
    val errorWi = error.data.zipWithIndex.grouped(_MSGGS)
    weightsWi.foreach {
      case (wi, j) => wi.foreach {
        d =>
          val b = WeightBatch(UUID.randomUUID.toString, job.id, d, j)
          _acks += b.id -> b
          sender ! b
      }
    }
    errorWi.foreach {
      d =>
        val b = ErrorBatch(UUID.randomUUID.toString, job.id, d)
        _acks += b.id -> b
        sender ! b
    }

  }

  protected def compute(xs: In, ys: Out, layers: Seq[Layer], weights: ArrayBuffer[DenseMatrix[Double]],
                      learningRate: Double, parallelism: Int): (Weights[Double], DenseMatrix[Double])


}
