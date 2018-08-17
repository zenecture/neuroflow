package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.DenseNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ShallowNet {

  /*

      This is a shallow neural net learning (0.3, 0.3) -> (0.5).

      The closed form solution is (-w1, +w2) or (+w1, -w2) where w1 = w2.

        Proof:   + w1*0.3 - w2*0.3 = 0.3w - 0.3w = 0
                 - w1*0.3 + w2*0.3 = 0.3w - 0.3w = 0
                                 =>  Sigmoid(0)  = 0.5 = y.

  */

  def apply = {

    implicit val weights = WeightBreeder[Double].random(-1, 1)

    val settings = Settings[Double](
      learningRate = { case (_, _) => 0.01 },
      precision = 1E-20,
      iterations = 100000
    )

    val net = Network(Vector(2) :: Dense(1, Sigmoid) :: SquaredError(), settings)

    net.train(Seq(->(0.3, 0.3)), Seq(->(0.5)))

    println("Output: " + net.evaluate(->(0.3, 0.3)))
    println("Parameters must be of shape: -a, +a or +a, -a")
    println("Network was: " + net)

  }

}

/*




                     _   __                      ________
                    / | / /__  __  ___________  / ____/ /___ _      __
                   /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                  / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
                 /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/
                                                                    1.7.4


                    Network : neuroflow.nets.cpu.DenseNetwork

                    Weights : 2 (≈ 1,52588e-05 MB)
                  Precision : Double

                       Loss : neuroflow.core.SquaredError
                     Update : neuroflow.core.Vanilla

                     Layout : 2 Vector
                              1 Dense (σ)






        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:19:574] Training with 1 samples, batch size = 1, batches = 1.
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:19:574] Breeding batches ...
        [scala-execution-context-global-64] DEBUG neuroflow.core.BatchBreeder$ - [17.08.2018 23:37:19:668] Bred Batch 0.
        Aug 17, 2018 11:37:19 PM com.github.fommil.jni.JniLoader liberalLoad
        INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader5479581012101716539netlib-native_system-osx-x86_64.jnilib
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:19:762] Iteration 1.1, Avg. Loss = 0,00416456, Vector: 0.004164557113913071
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:19:768] Iteration 2.1, Avg. Loss = 0,00416368, Vector: 0.004163681528061246
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:19:768] Iteration 3.1, Avg. Loss = 0,00416281, Vector: 0.004162806113613538
        ...
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:26:603] Iteration 99999.1, Avg. Loss = 7,53834e-13, Vector: 7.53833714358451E-13
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:26:603] Iteration 100000.1, Avg. Loss = 7,53664e-13, Vector: 7.53664111307632E-13
        [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [17.08.2018 23:37:26:603] Took 100000 of 100000 iterations.
        Output: DenseVector(0.5000012275948406)
        Parameters must be of shape: -a, +a or +a, -a
        Network was:
        ---
        -0.3423958262832021
        0.3424121942144106
        [success] Total time: 15 s, completed 17.08.2018 23:37:26


 */

