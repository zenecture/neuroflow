package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.application.processor.Image._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.application.plugin.Notation.Implicits.toVector
import neuroflow.nets.LBFGSNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    val groupSize = 1200 // max = 20*20*3 = 1200

    // Training
    val plus = extractRgb(getResourceFile("img/plus.png")).grouped(groupSize).toVector
    val plusRotated = extractRgb(getResourceFile("img/plus_rotated.png")).grouped(groupSize).toVector
    val heart = extractRgb(getResourceFile("img/heart.png")).grouped(groupSize).toVector
    val random = extractRgb(getResourceFile("img/random.png")).grouped(groupSize).toVector

    // Testing
    val heartDistorted = extractRgb(getResourceFile("img/heart_distorted.png")).grouped(groupSize).toVector
    val heartRotated = extractRgb(getResourceFile("img/heart_rotated.png")).grouped(groupSize).toVector

    println(s"Training ${plus.size + heart.size} samples...")

    val fn = Sigmoid
    val training = plus.zip(heart).zip(random).zip(plusRotated)
    val nets = training.par.map {
      case (((p, h), r), pr) =>
        val settings = Settings(iterations = 100)
        val net = Network(Input(p.size) :: Hidden(20, fn) :: Hidden(10, fn) :: Output(3, fn) :: HNil, settings)
        net.train(-->(p, h, r, pr), -->(->(1.0, 0.0, 0.0), ->(0.0, 1.0, 0.0), ->(0.0, 0.0, 1.0), ->(1.0, 0.0, 0.0)))
        net
    }

    def eval(image: Seq[Seq[Double]]) = {
      image.zip(nets).map { s =>
        val (xs, net) = s
        net.evaluate(xs)
      } reduce { (a, b) =>
        a.zip(b).map(l => l._1 + l._2)
      } map { end => end / nets.size }
    }

    val plusResult = eval(plus)
    val plusRotatedResult = eval(plusRotated)
    val heartResult = eval(heart)
    val heartDistortedResult = eval(heartDistorted)
    val heartRotatedResult = eval(heartRotated)
    val randomResult = eval(random)

    println(s"Plus classified: $plusResult")
    println(s"Plus Rotated classified: $plusRotatedResult")
    println(s"Heart classified: $heartResult")
    println(s"Heart distorted classified: $heartDistortedResult")
    println(s"Heart rotated classified: $heartRotatedResult")
    println(s"Random classified: $randomResult")

  }

}

/*

        General Layout:

          Plus ->(1.0, 0.0, 0.0)
          Heart ->(0.0, 1.0, 0.0)
          Random ->(0.0, 0.0, 1.0)

          [scala-execution-context-global-65] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 1,05332e-05 (rel: 0,460) 3,38469e-05
          [scala-execution-context-global-65] INFO neuroflow.nets.NFLBFGS - Step Size: 1,000
          [scala-execution-context-global-65] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 4,33024e-06 (rel: 0,589) 1,19527e-05
          [scala-execution-context-global-65] INFO neuroflow.nets.NFLBFGS - Converged because error function is sufficiently minimal.
          Plus classified: Vector(0.999999895367549, 0.0026118417961460503, 3.7604261592618936E-8)
          Plus Rotated classified: Vector(0.9999999295749643, 0.0021358566858272476, 2.2406596672020294E-8)
          Heart classified: Vector(0.001438279264291867, 0.9973241450760597, 0.0022719941679902405)
          Heart distorted classified: Vector(0.0014273469497369767, 0.9973195940390195, 0.002287412422013441)
          Heart rotated classified: Vector(0.006757400029251021, 0.5726952097176847, 0.0748294063598314)
          Random classified: Vector(8.187014938437293E-5, 4.43750291767083E-4, 0.9999415821496581)
          [success] Total time: 114 s, completed 19.04.2017 22:48:04

 */
