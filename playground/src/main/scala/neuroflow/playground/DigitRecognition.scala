package neuroflow.playground

import breeze.linalg.DenseVector
import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activators.Float._
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


  def digitSet2Vec(path: String): Seq[DenseVector[Float]] = {
    val selector: Int => Boolean = _ < 255
    (0 to 9) map (i => loadBinary(getResourceFile(path + s"$i.png"), selector).float)
  }

  def apply = {

    val config = (1 to 3).map(_ -> (0.0, 0.01)) :+ 4 -> (0.0, 0.1)
    implicit val weights = WeightBreeder[Float].normal(config.toMap)

    val sets = ('a' to 'h') map (c => digitSet2Vec(s"img/digits/$c/"))

    val xs = sets.dropRight(1).flatMap { s => (0 to 9).map { digit => s(digit) } }
    val ys = sets.dropRight(1).flatMap { m =>
      (0 to 9).map { digit =>
        val t = zero[Float](10)
        t.update(digit, 1.0f)
        t
      }
    }

    val (f, g) = (ReLU.biased(0.1f), ReLU.biased(1.0f))

    val L =
            Vector (xs.head.length)          ::
            Dense  (400, f)                  ::
            Dense  (200, f)                  ::
            Dense  (50, f)                   ::
            Dense  (10, g)                   ::   SoftmaxLogEntropy()

    val net = Network(
      layout = L,
      settings = Settings[Float](
        learningRate = { case (_, _) => 1E-4 },
        prettyPrint = true,
        updateRule = Momentum(0.8f),
        precision = 1E-3,
        iterations = 15000
      )
    )

    net.train(xs, ys)

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

    val setsResult = sets.map(s => s.map(v => net(v)))

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
                                                                    1.7.0


                    Network : neuroflow.nets.cpu.DenseNetwork

                    Weights : 170.500 (≈ 0,650406 MB)
                  Precision : Single

                       Loss : neuroflow.core.SoftmaxLogEntropy
                     Update : neuroflow.core.Momentum

                     Layout : 200 Vector
                              400 Dense (ReLU + Bias(0.1))
                              200 Dense (ReLU + Bias(0.1))
                              50 Dense (ReLU + Bias(0.1))
                              10 Dense (ReLU + Bias(1.0))






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



        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:18:887] Training with 70 samples, batch size = 70, batches = 1.
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:19:259] Breeding batches ...
        [scala-execution-context-global-67] DEBUG neuroflow.core.BatchBreeder$ - [13.08.2018 20:58:19:396] Bred Batch 0.
        Aug 13, 2018 8:58:19 PM com.github.fommil.jni.JniLoader liberalLoad
        INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader1733496608378989451netlib-native_system-osx-x86_64.jnilib
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:19:536] Iteration 1.1, Avg. Loss = 16,1312, Vector: 15.695352  16.388042  15.79654  16.736338  16.023125  17.038774  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:19:559] Iteration 2.1, Avg. Loss = 16,1311, Vector: 15.695781  16.387875  15.796873  16.735378  16.02314  17.037626  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:19:569] Iteration 3.1, Avg. Loss = 16,1310, Vector: 15.696555  16.387573  15.797471  16.733648  16.023165  ... (10 total)
        ...
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:36:281] Iteration 2513.1, Avg. Loss = 0,0100014, Vector: 0.009870188  0.012217878  0.016114127  0.0054787886  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:36:288] Iteration 2514.1, Avg. Loss = 0,00999423, Vector: 0.0098633645  0.012208382  0.01610323  0.0054752673  ... (10 total)
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkFloat - [13.08.2018 20:58:36:288] Took 2514 of 15000 iterations.
        Pos: 100289, Neg: 70211
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
        [success] Total time: 21 s, completed 13.08.2018 20:58:36



 */