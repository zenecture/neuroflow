package neuroflow.playground

import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.Notation._
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._

/**
  * @author bogdanski
  * @since 03.01.16
  */


object AgeEarnings {

  /*

     Here we compare Neural Net vs. Gaussian.
     Feel free to read this article for the full story:
        http://znctr.com/blog/gaussian-vs-neural-net

  */


  def apply = {

    val train = scala.io.Source.fromFile(getResourceFile("file/income.txt")).getLines.map(_.split(",")).flatMap { k =>
      (if (k.length > 14) Some(k(14)) else None).map { over50k => (k(0).toDouble, if (over50k.equals(" >50K")) 1.0 else 0.0) }
    }.toArray

    val sets = Settings[Double](
      learningRate = { case (_, _) => 1E-3 },
      batchSize = Some(2000),
      precision = 1E-3,
      iterations = 200000)

    import neuroflow.nets.gpu.DenseNetwork._
    implicit val breeder = neuroflow.core.WeightBreeder[Double].random(-1, 1)

    val network = Network(Vector(1) :: Dense(20, Sigmoid) :: Dense(1, Sigmoid) :: SquaredMeanError(), sets)

    val maxAge = train.map(_._1).sorted.reverse.head
    val xs = train.map(a => ->(a._1 / maxAge))
    val ys = train.map(a => ->(a._2))
    network.train(xs, ys)

    val allOver = train.filter(_._2 == 1.0)
    val ratio = allOver.length / train.length
    val mean = allOver.map(_._1).sum / allOver.length

    println(s"Mean of all $mean")
    println(s"Ratio $ratio")

    val result = Range.Double(0.0, 1.1, 0.1).map(k => (k * maxAge, network(->(k))))
    val sum = result.map(_._2.apply(0)).sum
    println("Age, earning >50K")
    result.foreach { r => println(s"${r._1}, ${r._2(0) * (1 / sum)}")} // normalized probabilities p such that Σp = 1.0

  }


  /*






                   _   __                      ________
                  / | / /__  __  ___________  / ____/ /___ _      __
                 /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
               /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


                  Version : 1.4.0

                  Network : neuroflow.nets.gpu.DenseNetwork
                     Loss : neuroflow.core.SquaredMeanError
                   Update : neuroflow.core.Vanilla

                   Layout : 1 In
                            20 Dense (σ)
                            1 Out (σ)

                  Weights : 40 (≈ 0,000305176 MB)
                Precision : Double




                     O
                     O
                     O
                     O
                     O
               O     O     O
                     O
                     O
                     O
                     O



      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 15:23:11:289] Loading module: matrix_kernels_float.ptx
      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 15:23:11:393] Loading module: matrix_kernels_double.ptx
      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 15:23:11:408] Loading module: matrix_convops_float.ptx
      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 15:23:11:413] Loading module: matrix_convops_double.ptx
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 15:23:12:096] Training with 32561 samples, batch size = 2000, batches = 17 ...
      [run-main-0] DEBUG neuroflow.cuda.GcThreshold$ - [20.01.2018 15:23:12:207] GcThreshold = 704960 Bytes (≈ 0,672302 MB)
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 15:23:12:483] Iteration 1.1, Avg. Loss = 444,220, Vector: 444.2197471042573
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 15:23:12:507] Iteration 2.2, Avg. Loss = 265,925, Vector: 265.92520111270863
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 15:23:12:527] Iteration 3.3, Avg. Loss = 189,556, Vector: 189.5559280688758
      ...
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 16:06:02:207] Iteration 199999.11, Avg. Loss = 162,881, Vector: 162.8806747880057
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 16:06:02:218] Iteration 200000.12, Avg. Loss = 167,398, Vector: 167.39834480268837
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 16:06:02:222] Took 200000 of 200000 iterations.
      Mean of all 44.24984058155847
      Ratio 0
      Age, earning >50K
      0.0, 3.843528892275602E-7
      9.0, 7.318520013036975E-5
      18.0, 0.004528900299998429
      27.0, 0.055430214357576356
      36.0, 0.15949873792409597
      45.0, 0.21053139597387885
      54.0, 0.19897587126743807
      63.0, 0.15578791522395138
      72.0, 0.10695089167732393
      81.0, 0.06743998317560304
      90.0, 0.040782520547114336
      [success] Total time: 2588 s, completed 20.01.2018 16:06:04

   */
}
