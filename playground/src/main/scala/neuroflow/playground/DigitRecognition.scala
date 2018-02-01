package neuroflow.playground

import breeze.linalg.DenseVector
import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activator.ReLU
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.DenseNetwork._

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


  def digitSet2Vec(path: String): Seq[DenseVector[Double]] = {
    val selector: Int => Boolean = _ < 255
    (0 to 9) map (i => extractBinary(getResourceFile(path + s"$i.png"), selector))
  }

  def apply = {

    val config = (0 to 2).map(_ -> (0.01, 0.01)) :+ 3 -> (0.1, 0.1)
    implicit val wp = neuroflow.core.WeightProvider[Float].normal(config.toMap)

    val sets = ('a' to 'h') map (c => digitSet2Vec(s"img/digits/$c/"))

    val xs = sets.dropRight(1).flatMap { s => (0 to 9).map { digit => s(digit).map(_.toFloat) } }
    val ys = sets.dropRight(1).flatMap { m => (0 to 9).map { digit =>
      ->(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).toScalaVector.updated(digit, 1.0).denseVec.map(_.toFloat) }
    }

    val fn = ReLU

    val net = Network(
      layout =
         Vector (xs.head.length)  ::
         Dense  (400, fn)         ::
         Dense  (200, fn)         ::
         Dense  (50, fn)          ::
         Dense  (10, fn)          ::  Softmax(),
      settings = Settings[Float](
        learningRate = { case (_, _) => 1E-5 },
        updateRule = Momentum(0.8f),
        precision = 1E-3,
        iterations = 15000
      )
    )

    net.train(xs, ys)

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

    val setsResult = sets.map(s => s.map(v => net(v.float)))

    ('a' to 'h') zip setsResult foreach {
      case (char, res) =>
        ~> (println(s"set $char:")) next (0 to 9) foreach { digit =>
          println(s"$digit classified as " + res(digit).toScalaVector.indexOf(res(digit).max))
        }
    }

  }

}

/*






                     _   __                      ________
                    / | / /__  __  ___________  / ____/ /___ _      __
                   /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                  / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
                 /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


                    Version : 1.3.4

                    Network : neuroflow.nets.cpu.DenseNetwork
                       Loss : neuroflow.core.Softmax
                     Update : neuroflow.core.Momentum

                     Layout : 200 In
                              400 Dense (R)
                              200 Dense (R)
                              50 Dense (R)
                              10 Out (R)

                    Weights : 170.500 (≈ 0,650406 MB)
                  Precision : Single




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



        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:26:824] Training with 70 samples, batch size = 70, batches = 1 ...
        Dez 14, 2017 5:03:26 PM com.github.fommil.jni.JniLoader liberalLoad
        INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader7879827578054582548netlib-native_system-osx-x86_64.jnilib
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:090] Iteration 1 - Loss 0,842867 - Loss Vector 1.3594189  1.0815932  0.06627092  1.2282351  0.40324837  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:152] Iteration 2 - Loss 0,760487 - Loss Vector 1.2405235  1.0218079  0.080294095  1.1119698  0.30945536  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:168] Iteration 3 - Loss 0,631473 - Loss Vector 1.0470318  0.9293458  0.10742891  0.92295074  0.16466755  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:178] Iteration 4 - Loss 0,528371 - Loss Vector 0.8682083  0.8433325  0.17737864  0.7455  0.062339883  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:188] Iteration 5 - Loss 0,439425 - Loss Vector 0.6963819  0.7676125  0.23556742  0.5764897  0.062838934  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:197] Iteration 6 - Loss 0,373034 - Loss Vector 0.5466118  0.70639104  0.2790047  0.43036366  0.1531757  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:205] Iteration 7 - Loss 0,328959 - Loss Vector 0.42401966  0.65744436  0.3100834  0.31236598  0.25140944  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:213] Iteration 8 - Loss 0,289892 - Loss Vector 0.30970943  0.61508274  0.32081065  0.20909563  0.32359478  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:03:27:220] Iteration 9 - Loss 0,290616 - Loss Vector 0.248706  0.5873246  0.34531453  0.16553785  0.39594984  ... (10 total)
        ...
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:04:25:607] Iteration 15000 - Loss 0,000370492 - Loss Vector 2.0882876E-4  3.3544307E-4  3.2464182E-4  2.5632523E-4  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:04:25:607] Took 15000 iterations of 15000 with Loss = 0,000370492
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkSingle - [14.12.2017 17:04:25:608] Applying KeepBest strategy. Best test error so far: 0,000370492.
        Pos: 128063, Neg: 42437
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
        [success] Total time: 70 s, completed 14.12.2017 17:04:26

 */