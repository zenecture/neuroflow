package neuroflow.nets.distributed

import java.util.UUID

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import breeze.linalg.DenseMatrix
import com.typesafe.config.ConfigFactory
import neuroflow.common.Logs
import neuroflow.core.Network.{Matrix, _}
import neuroflow.core.{HasActivator, Layer, Node, Settings}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.duration._
import scala.concurrent.forkjoin.ForkJoinPool
import scala.util.{Failure, Success}

/**
  * @author bogdanski
  * @since 28.08.17
  */

object DefaultExecutor extends Logs {
  def apply(node: Node, xs: Vectors, ys: Vectors, settings: Settings = Settings()): Unit = {

    info(s"Booting DefaultExecutor ${node.host}:${node.port} ...")

    val _akka = ActorSystem("NeuroFlow", ConfigFactory.parseString(
      s"""
         |akka {
         |  log-dead-letters = 0
         |  extensions = ["com.romix.akka.serialization.kryo.KryoSerializationExtension$$"]
         |  actor {
         |    provider = remote
         |    kryo {
         |      type = "nograph"
         |      idstrategy = "incremental"
         |      implicit-registration-logging = true
         |    }
         |    serializers {
         |      kryo = "com.twitter.chill.akka.AkkaSerializer"
         |    }
         |    serialization-bindings {
         |      "neuroflow.nets.distributed.Message" = kryo
         |    }
         |  }
         |  remote {
         |    artery {
         |      enabled = on
         |      canonical.hostname = "${node.host}"
         |      canonical.port = ${node.port}
         |      advanced {
         |        maximum-frame-size = ${settings.transport.frameSize}
         |        maximum-large-frame-size = ${settings.transport.frameSize}
         |      }
         |    }
         |  }
         |}
      """.stripMargin))

    _akka.actorOf(Props(new DefaultExecutor(xs, ys, settings)), "executor")

    info("Up and running.")
    info("Type 'exit' to exit.")

    while (scala.io.StdIn.readLine() != "exit") { }

    import _akka.dispatcher

    _akka.terminate().onComplete {
      case Success(_) => System.exit(0)
      case Failure(_) => System.exit(1)
    }

  }
}

class DefaultExecutor(xs: Vectors, ys: Vectors, settings: Settings) extends Actor with Logs {

  import context.dispatcher

  private val _scheduler  = context.system.scheduler
  private var _resender   = _scheduler.schedule(1 minute, 10 seconds, self, 'Resend)

  private val _MSGGS      = settings.transport.messageGroupSize

  private val _weights    = ArrayBuffer.empty[Matrix]
  private var _job: Job   = _
  private var _batchCount = 0
  private val _buffer     = ArrayBuffer.empty[WeightBatch]

  private val _acks       = collection.mutable.HashMap.empty[String, Message]
  private var _request: ActorRef = _

  def receive = {

    case Heartbeat =>
      info(s"Connected to $sender.")
      sender ! Heartbeat

    case job @ Job(id, _, _, weightDims, _, _) =>
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

    case batch @ WeightBatch(id, jobId, _, _) if _job == null =>
      debug(s"Got batch, but no job for id = $jobId. Buffering ...")
      sender ! Ack(id)
      _buffer += batch
      _scheduler.scheduleOnce(3 seconds, self, 'ReprocessBuffer)

    case b @ WeightBatch(id, jobId, _, position) if position < 0 =>
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
        val in = xs.map(x => x.asDenseMatrix)
        val out = ys.map(y => y.asDenseMatrix)
        val (weights, error) = compute(in, out, _job.layers, _weights, _job.learningRate, _job.parallelism)
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

  private def sendResults(job: Job, weights: Weights, error: Matrix, sender: ActorRef): Unit = {
    val weightsWi = weights.map(_.data.zipWithIndex.grouped(_MSGGS)).zipWithIndex
    val errorWi   = error.data.zipWithIndex.grouped(_MSGGS)
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

  private def compute(xs: Matrices, ys: Matrices, layers: Seq[Layer], weights: ArrayBuffer[Matrix],
                      learningRate: Double, parallelism: Int): (Weights, Matrix) = {

    val _xsys = xs.par.zip(ys)
    _xsys.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(parallelism))

    val _layersNI       = layers.tail.map { case h: HasActivator[Double] => h }
    val _outputDim      = layers.last.neurons
    val _lastWlayerIdx  = weights.size - 1

    val _ds = (0 to _lastWlayerIdx).map { i =>
      i -> DenseMatrix.zeros[Double](weights(i).rows, weights(i).cols)
    }.toMap

    val _errSum = DenseMatrix.zeros[Double](1, _outputDim)
    val _square = DenseMatrix.zeros[Double](1, _outputDim)
    _square := 2.0

    _xsys.map { xy =>
      val (x, y) = xy
      val fa  = collection.mutable.Map.empty[Int, Matrix]
      val fb  = collection.mutable.Map.empty[Int, Matrix]
      val dws = collection.mutable.Map.empty[Int, Matrix]
      val ds  = collection.mutable.Map.empty[Int, Matrix]
      val e   = DenseMatrix.zeros[Double](1, _outputDim)

      @tailrec def forward(in: Matrix, i: Int): Unit = {
        val p = in * weights(i)
        val a = p.map(_layersNI(i).activator)
        val b = p.map(_layersNI(i).activator.derivative)
        fa += i -> a
        fb += i -> b
        if (i < _lastWlayerIdx) forward(a, i + 1)
      }

      @tailrec def derive(i: Int): Unit = {
        if (i == 0 && _lastWlayerIdx == 0) {
          val yf = y - fa(0)
          val d = -yf *:* fb(0)
          val dw = x.t * d
          dws += 0 -> dw
          e += yf
        } else if (i == _lastWlayerIdx) {
          val yf = y - fa(i)
          val d = -yf *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          e += yf
          derive(i - 1)
        } else if (i < _lastWlayerIdx && i > 0) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = fa(i - 1).t * d
          dws += i -> dw
          ds += i -> d
          derive(i - 1)
        } else if (i == 0) {
          val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
          val dw = x.t * d
          dws += i -> dw
        }
      }

      forward(x, 0)
      derive(_lastWlayerIdx)
      e :^= _square
      e *= 0.5
      (dws, e)
    }.seq.foreach { ab =>
      _errSum += ab._2
      var i = 0
      while (i <= _lastWlayerIdx) {
        val m = _ds(i)
        val n = ab._1(i)
        m += (n *= learningRate)
        i += 1
      }
    }

    (_ds.values.toIndexedSeq, _errSum)

  }

}
