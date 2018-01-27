package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ShallowWeights {

  /*

      This is a shallow neural net learning (0.3, 0.3) -> (0.5).

      The closed form solution is (-w1, +w2) or (+w1, -w2) where w1 = w2.

        Proof:    w1*0.3 + (-w2*0.3) = 0.3w - 0.3w = 0
                     =>  Sigmoid(0)  = 0.5 = y.

      The goal is to find weights close to this exact shape.

  */

  def apply = {

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    val settings = Settings[Double](
      learningRate = { case (_, _) => 0.01 },
      precision = 1E-20,
      iterations = 100000
    )

    val net = Network(Input(2) :: Dense(1, Sigmoid) :: SquaredMeanError(), settings)

    net.train(Seq(->(0.3, 0.3)), Seq(->(0.5)))

    println("Output: " + net.evaluate(->(0.3, 0.3)))
    println("Parameters must roughly be of shape: -a, +a or +a, -a")
    println("Network was: " + net)

  }

}

/*


                 _   __                      ________
                / | / /__  __  ___________  / ____/ /___ _      __
               /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
              / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
             /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


                Version : 1.4.0

                Network : neuroflow.nets.cpu.DenseNetwork
                   Loss : neuroflow.core.Softmax
                 Update : neuroflow.core.Vanilla

                 Layout : 2 In
                          1 Out (σ)

                Weights : 2 (≈ 1,52588e-05 MB)
              Precision : Double




             O
             O     O



    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [25.01.2018 13:08:39:494] Training with 1 samples, batch size = 1, batches = 1 ...
    Jan 25, 2018 1:08:39 PM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader3459823189525380021netlib-native_system-osx-x86_64.jnilib
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [25.01.2018 13:08:39:674] Iteration 1.1, Avg. Loss = 0,346574, Vector: 0.34657359027997264
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [25.01.2018 13:08:39:678] Iteration 2.1, Avg. Loss = 0,346574, Vector: 0.34657359027997264
    ...
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [25.01.2018 13:09:20:586] Iteration 99999.1, Avg. Loss = 3,10725e-13, Vector: 3.1072471834145533E-13
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [25.01.2018 13:09:20:586] Iteration 100000.1, Avg. Loss = 3,10655e-13, Vector: 3.106548091796204E-13
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [25.01.2018 13:09:20:586] Took 100000 of 100000 iterations.
    Output: DenseVector(0.4999992118567189)
    Parameters must roughly be of shape: -a, +a or +a, -a
    Network was:
    ---
    -0.5421605034076382
    0.5421499948305565


 */
