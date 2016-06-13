package neuroflow.playground

import neuroflow.application.plugin.Style._
import neuroflow.application.preprocessor.Util._
import neuroflow.application.preprocessor.Image._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core._
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
    val plus = extractRgb(getFile("img/plus.png")).grouped(groupSize).toList
    val heart = extractRgb(getFile("img/heart.png")).grouped(groupSize).toList

    // Testing
    val heartDistorted = extractRgb(getFile("img/heart_distorted.png")).grouped(groupSize).toList
    val heartRotated = extractRgb(getFile("img/heart_rotated.png")).grouped(groupSize).toList
    val plusRotated = extractRgb(getFile("img/plus_rotated.png")).grouped(groupSize).toList
    val random = extractRgb(getFile("img/random.png")).grouped(groupSize).toList

    println(s"Training ${plus.size + heart.size} samples...")

    val fn = Sigmoid
    val training = plus.zip(heart).zip(random).zip(plusRotated)
    val nets = training.par.map {
      case (((p, h), r), pr) =>
        val settings = Settings(maxIterations = 100)
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


        [ForkJoinPool-1-worker-11] INFO neuroflow.nets.NFLBFGS - Val and Grad Norm: 4,89532e-06 (rel: 0,650) 1,52269e-05
        [ForkJoinPool-1-worker-11] INFO neuroflow.nets.NFLBFGS - Converged because error function is sufficiently minimal.
        Plus classified: Vector(0.9999999999998685, 7.429070100868865E-4, 5.532277500786511E-13)
        Plus Rotated classified: Vector(0.9999999999998996, 0.0031738983574836634, 1.0308867367481122E-13)
        Heart classified: Vector(0.0017272186991263289, 0.9966483389358642, 2.878172153766738E-4)
        Heart distorted classified: Vector(2.4062400408535897E-6, 0.962381573185858, 0.4671002608813794)
        Heart rotated classified: Vector(1.8880785792863597E-6, 0.9479183304780576, 0.5687777847544573)
        Random classified: Vector(1.0617016570154908E-7, 0.002106536353694731, 0.9999043998116915)
        [success] Total time: 137 s, completed 13.06.2016 23:16:30

 */
