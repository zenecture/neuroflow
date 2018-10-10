package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.ConvNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object XORConv {

  def apply = {

    /*

       The XOR-function is linearly not separable, so we need
       something which naturally copes with non-linearities.

       The dense XOR expressed using a convolution.

     */

    implicit val weights = WeightBreeder[Double].normal(μ = 0.0, σ = 1.0)

    val f = Sigmoid

    val c = Convolution(
      dimIn = (1, 2, 1),
      padding = (0, 0),
      field = (1, 2),
      stride = (1, 1),
      filters = 3,
      activator = f
    )

    val L = c :: Dense(1, f) :: SquaredError()

    val net = Network(
      layout = L,
      settings = Settings[Double](
        learningRate = { case (_, _) => 1.0 },
        precision = 1E-5,
        iterations = 100000
      )
    )

    val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0)).map(v => Tensor3D.fromVector(v))
    val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))

    net.train(xs, ys)

    xs.foreach { x =>
      println(s"Input: $x, Output: ${net(x)}")
    }

    println("Network was: " + net)

  }

}


/*






                     _   __                      ________
                    / | / /__  __  ___________  / ____/ /___ _      __
                   /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                  / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
                 /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/
                                                                    1.7.8


                    Network : neuroflow.nets.cpu.ConvNetwork

                    Weights : 9 (≈ 6,86646e-05 MB)
                  Precision : Double

                       Loss : neuroflow.core.SquaredError
                     Update : neuroflow.core.Vanilla

                     Layout : 1*2*1 ~> [1*2 : 1*1] ~> 1*1*3 (σ)
                              1 Dense (σ)






        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:12:126] Training with 4 samples, batch size = 4, batches = 1.
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:12:161] Breeding batches ...
        [scala-execution-context-global-98] DEBUG neuroflow.core.BatchBreeder$ - [10.10.2018 19:57:12:166] Bred Batch 0.
        Okt 10, 2018 7:57:12 PM com.github.fommil.jni.JniLoader liberalLoad
        INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader5056365765405121090netlib-native_system-osx-x86_64.jnilib
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:12:234] Iteration 1.1, Avg. Loss = 0,870768, Vector: 0.8707682296146382
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:12:239] Iteration 2.1, Avg. Loss = 0,857056, Vector: 0.8570558665294984
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:12:240] Iteration 3.1, Avg. Loss = 0,841429, Vector: 0.8414288499774292
        ...
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:269] Iteration 99995.1, Avg. Loss = 8,69992e-05, Vector: 8.69992150706616E-5
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:269] Iteration 99996.1, Avg. Loss = 8,69983e-05, Vector: 8.69983330485415E-5
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:269] Iteration 99997.1, Avg. Loss = 8,69975e-05, Vector: 8.69974510442554E-5
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:269] Iteration 99998.1, Avg. Loss = 8,69966e-05, Vector: 8.699656905780086E-5
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:269] Iteration 99999.1, Avg. Loss = 8,69957e-05, Vector: 8.699568708917678E-5
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:269] Iteration 100000.1, Avg. Loss = 8,69948e-05, Vector: 8.699480513838167E-5
        [run-main-1] INFO neuroflow.nets.cpu.ConvNetworkDouble - [10.10.2018 19:57:18:270] Took 100000 of 100000 iterations.
        Input: <function1>, Output: DenseVector(0.010011197119973924)
        Input: <function1>, Output: DenseVector(0.9940019342268774)
        Input: <function1>, Output: DenseVector(0.993995487353722)
        Input: <function1>, Output: DenseVector(0.0013163637407054751)
        Network was:
        ---
        7.564896558018516   -4.9449775557587765
        5.977508776503334   5.8571463815409075
        -4.690517285587798  6.792112133654039
        ---
        -14.440366052745667
        19.723647293494775
        -14.471260149271036


 */

