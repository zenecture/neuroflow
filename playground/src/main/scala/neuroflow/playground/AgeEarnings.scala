package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.core.Activator._
import neuroflow.core._
import shapeless._


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

    val src = scala.io.Source.fromFile(getResourceFile("file/income.txt")).getLines.map(_.split(",")).flatMap { k =>
      (if (k.length > 14) Some(k(14)) else None).map { over50k => (k(0).toDouble, if (over50k.equals(" >50K")) 1.0 else 0.0) }
    }.toArray

    val train = src
    val sets = Settings[Double](learningRate = { case (_, _) => 1E-3 }, batchSize = Some(2000), precision = 1E-3, iterations = 200000)

    import neuroflow.nets.gpu.DenseNetwork._
    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    val network = Network(Input(1) :: Dense(20, Sigmoid) :: Output(1, Sigmoid) :: HNil, sets)

    val maxAge = train.map(_._1).sorted.reverse.head
    val xs = train.map(a => ->(a._1 / maxAge))
    val ys = train.map(a => ->(a._2))
    network.train(xs, ys)

    val allOver = src.filter(_._2 == 1.0)
    val ratio = allOver.length / src.length
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



      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 11:02:05:580] Loading module: matrix_kernels_float.ptx
      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 11:02:05:654] Loading module: matrix_kernels_double.ptx
      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 11:02:05:661] Loading module: matrix_convops_float.ptx
      [run-main-0] DEBUG neuroflow.cuda.CuMatrix$ - [20.01.2018 11:02:05:664] Loading module: matrix_convops_double.ptx
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 11:02:06:153] Training with 32561 samples, batch size = 32561, batches = 1 ...
      [run-main-0] DEBUG neuroflow.cuda.GcThreshold$ - [20.01.2018 11:02:08:235] GcThreshold = 11462432 Bytes (≈ 10,9314 MB)
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 11:02:08:459] Iteration 1. Loss Ø: 3111,91 Σ: 3111.90871853784
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 11:02:08:611] Iteration 2. Loss Ø: 3801,08 Σ: 3801.0765619411727
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 11:02:08:737] Iteration 3. Loss Ø: 3709,15 Σ: 3709.1531637151556
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 11:02:08:852] Iteration 4. Loss Ø: 3416,37 Σ: 3416.368354038494
      ...
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 13:37:09:568] Iteration 99999. Loss Ø: 2664,91 Σ: 2664.9144190131383
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 13:37:09:656] Iteration 100000. Loss Ø: 2664,91 Σ: 2664.914417503272
      [run-main-0] INFO neuroflow.nets.gpu.DenseNetworkDouble - [20.01.2018 13:37:09:656] Took 100000 of 100000 iterations.
      Mean of all 44.24984058155847
      Ratio 0
      Age, earning >50K
      0.0, 5.248582246681653E-7
      9.0, 8.556673875045431E-5
      18.0, 0.004777585359808839
      27.0, 0.05584745691881924
      36.0, 0.15894439896795678
      45.0, 0.209808291589738
      54.0, 0.19865528233360355
      63.0, 0.15589031630756764
      72.0, 0.10726154035035684
      81.0, 0.06774443527175603
      90.0, 0.04098460130341816
      [success] Total time: 9249 s, completed 20.01.2018 13:37:10

   */
}
