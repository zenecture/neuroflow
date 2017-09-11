package neuroflow.nets.distributed

import java.util.UUID

import akka.actor.{Actor, ActorRef, ActorSelection, ActorSystem, PoisonPill, Props}
import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.typesafe.config.ConfigFactory
import neuroflow.common.Logs
import neuroflow.core._
import neuroflow.nets.Registry

import scala.annotation.tailrec
import scala.collection.Seq
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}

/**
  * @author bogdanski
  * @since 11.09.17
  */
class ProcessorActor(x: ActorSelection, stepSize: Double, _weightsRoCo: IndexedSeq[(Int, Int)],
                     _outputDim: Int, _weightsWi: IndexedSeq[(Iterator[Array[(Double, Int)]], Int)],
                     layers: Seq[Layer], settings: Settings) extends Actor with Logs {

  import context.dispatcher

  private val _weights = _weightsRoCo.map {
    case (rows, cols) => DenseMatrix.create[Double](rows, cols, Array.fill(rows * cols)(0.0))
  }
  private val _error   = DenseMatrix.zeros[Double](1, _outputDim)
  private val _id      = UUID.randomUUID.toString
  private val _batches = _weightsWi.flatMap {
    case (wi, j) => wi.map { d => WeightBatch(UUID.randomUUID.toString, _id, d, j) }
  }

  private val _scheduler  = context.system.scheduler

  private val _acks       = collection.mutable.HashMap.empty[String, Message]

  private val _batchC_T   = _batches.length
  private val _batchE_T   = _error.data.zipWithIndex.grouped(settings.transport.messageGroupSize).size
  private var _batchC     = 0
  private var _batchE     = 0

  private var _request: ActorRef = _

  def receive: Receive = {
    case 'Execute =>
      _request = sender()
      _scheduler.schedule(1 minute, 10 seconds, self, 'Resend)
      x ! Job(_id, _batches.length, layers, _weightsRoCo, stepSize, settings.parallelism)
      _batches.foreach { wb =>
        _acks += wb.id -> wb
        x ! wb
      }

    case 'Resend =>
      debug(s"Resending ${_acks.size} batches ...")
      _acks.values.foreach(x ! _)

    case Ack(id) =>
      debug(s"Got ack $id")
      _acks -= id

    case b @ WeightBatch(id, jobId, _, position) if position < 0 =>
      debug(s"Got malformed WeightBatch $b. Discarding it ...")

    case WeightBatch(id, jobId, data, position) =>
      debug(s"Got WeightBatch for jobId = $jobId")
      sender ! Ack(id)
      _batchC += 1
      data.foreach { case (v, i) => _weights(position).data.update(i, v) }
      maybeComplete()

    case ErrorBatch(id, jobId, data) =>
      debug(s"Got ErrorBatch for jobId = $jobId")
      sender ! Ack(id)
      _batchE += 1
      data.foreach { case (v, i) => _error.data.update(i, v) }
      maybeComplete()

  }

  private def maybeComplete(): Unit =
    if ((_batchC >= _batchC_T) && (_batchE >= _batchE_T)) {
      _request ! (_weights, _error)
      self ! PoisonPill
    }

}
