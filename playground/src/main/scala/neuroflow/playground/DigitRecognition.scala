package neuroflow.playground

import breeze.linalg.DenseVector
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.application.processor.Util._
import neuroflow.common.VectorTranslation._
import neuroflow.common.~>
import neuroflow.core.Activator.ReLU
import neuroflow.core._
import neuroflow.nets.gpu.DenseNetwork._
import shapeless._

import scala.collection.immutable

/**
  * @author bogdanski
  * @since 04.01.16
  */
object DigitRecognition {

  /*

   Here the goal is to classify digits from unknown font family 'h'.
   Feel free to read this article for the full story:
      http://znctr.com/blog/digit-recognition

  */


  def getDigitSet(path: String): immutable.IndexedSeq[scala.Vector[Array[Double]]] = {
    val selector: Int => Boolean = _ < 255
    (0 to 9) map (i => extractBinary(getResourceFile(path + s"$i.png"), selector).data.grouped(200).toVector)
  }

  def apply = {

    val config = (0 to 2).map(_ -> (0.01, 0.01)) :+ 3 -> (0.1, 0.1)
    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].normal(config.toMap)

    val sets = ('a' to 'h') map (c => getDigitSet(s"img/digits/$c/").toVector)
    val nets = sets.head.head.indices.par.map { segment =>

      val xs = sets.dropRight(1).flatMap { s => (0 to 9).map { digit => s(digit)(segment) } }.toArray
      val ys = sets.dropRight(1).flatMap { m => (0 to 9).map { digit => ->(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).toScalaVector.updated(digit, 1.0) } }.toArray


      val fn = ReLU

      val settings = Settings(
        learningRate = { case (_, _) => 1E-5 },
        updateRule = Momentum(0.8),
        lossFunction = Softmax(),
        precision = 1E-4,
        prettyPrint = true, iterations = 15000,
        regularization = Some(KeepBest))

      val net = Network(
           Input(xs.head.size)   ::
           Dense(400, fn)        ::
           Dense(200, fn)        ::
           Dense(50, fn)         ::
           Output(10, fn)        ::  HNil, settings)

      net.train(xs.map(l => DenseVector(l)), ys.map(_.dv))

      val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
      val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

      println(s"Pos: $posWeights, Neg: $negWeights")

      net

    }

    val setsResult = sets map { set => set map { d => d flatMap { xs => nets map { _.evaluate(DenseVector(xs)) } } reduce(_ + _) map (end => end / nets.size) } }

    ('a' to 'h') zip setsResult foreach {
      case (char, res) =>
        ~> (println(s"set $char:")) next (0 to 9) foreach { digit => println(s"$digit classified as " + res(digit).toScalaVector.indexOf(res(digit).max)) }
    }

  }

}

/*





                     _   __                      ________
                    / | / /__  __  ___________  / ____/ /___ _      __
                   /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                  / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
                 /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


                    Version : 1.2.4
                    Network : neuroflow.nets.cpu.DenseNetwork
                       Loss : neuroflow.core.Softmax

                     Layout : 200 In
                              400 Dense (R)
                              200 Dense (R)
                              50 Dense (R)
                              10 Out (R)

                    Weights : 170.500 (≈ 1,30081 MB)
                  Precision : Double




                       O
                       O
                 O     O     O
                 O     O     O
                 O     O     O     O     O
                 O     O     O     O     O
                 O     O     O
                 O     O     O
                       O
                       O



        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:08:847] Training with 70 samples, batchize = 70 ...
        Okt 03, 2017 7:00:08 PM com.github.fommil.jni.JniLoader liberalLoad
        INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader2845731484237263048netlib-native_system-osx-x86_64.jnilib
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:064] Iteration 1 - Loss 0,566935 - Loss Vector 0.06797198067471424  0.3917490757095061  1.2132633531520545  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:209] Iteration 2 - Loss 0,511307 - Loss Vector 0.08499523032105315  0.35229324500325926  1.135623026717559  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:245] Iteration 3 - Loss 0,432285 - Loss Vector 0.12251421015227973  0.2941788377991336  1.016693403844844  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:282] Iteration 4 - Loss 0,364590 - Loss Vector 0.18401166798151963  0.23872347778994105  0.8947084502385266  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:318] Iteration 5 - Loss 0,317478 - Loss Vector 0.2517408863639056  0.1981951408905487  0.7853158346366302  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:349] Iteration 6 - Loss 0,300827 - Loss Vector 0.3272640018536161  0.18281651866851517  0.6994208759998832  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:386] Iteration 7 - Loss 0,295964 - Loss Vector 0.3864266942021326  0.1877941838305194  0.6213750503074743  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:432] Iteration 8 - Loss 0,287068 - Loss Vector 0.41300497483783577  0.20211363120302944  0.5380031384571229  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:493] Iteration 9 - Loss 0,276471 - Loss Vector 0.4139268066586661  0.22046959850310044  0.4522435106791206  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:553] Iteration 10 - Loss 0,267513 - Loss Vector 0.3968239719463647  0.23979640970927787  0.36857703338881476  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:00:09:639] Iteration 11 - Loss 0,257345 - Loss Vector 0.3613530116752733  0.256995592325175  0.2871951551539933  ... (10 total)
        ...
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:08:33:528] Iteration 14998 - Loss 0,000305330 - Loss Vector 1.979127211258661E-4  3.269643859006314E-4  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:08:33:570] Iteration 14999 - Loss 0,000305285 - Loss Vector 1.9788607092391852E-4  3.2691588114313095E-4  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:08:33:605] Iteration 15000 - Loss 0,000305239 - Loss Vector 1.9785950144915357E-4  3.268669468383327E-4  ... (10 total)
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:08:33:605] Took 15000 iterations of 15000 with Loss = 0,000305239
        [scala-execution-context-global-66] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.10.2017 19:08:33:605] Applying KeepBest strategy. Best test error so far: 0,000305239.
        Pos: 128444, Neg: 42056
        set a:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set b:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set c:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set d:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set e:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set f:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set g:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        set h:
        0 classified as 0
        1 classified as 1
        2 classified as 2
        3 classified as 3
        4 classified as 4
        5 classified as 5
        6 classified as 6
        7 classified as 7
        8 classified as 8
        9 classified as 9
        [success] Total time: 513 s, completed 03.10.2017 19:08:34

 */