package neuroflow.playground

import neuroflow.application.plugin.Notation.Implicits.toVector
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.application.processor.Util._
import neuroflow.common.VectorTranslation._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
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
        val settings = Settings(iterations = 1000, learningRate = { case i if i < 100 => 0.5 case _ => 0.1 })
        val net = Network(Input(p.size) :: Hidden(20, fn) :: Hidden(10, fn) :: Output(3, fn) :: HNil, settings)
        net.train(Seq(p.dv, h.dv, r.dv, pr.dv), Seq(->(1.0, 0.0, 0.0), ->(0.0, 1.0, 0.0), ->(0.0, 0.0, 1.0), ->(1.0, 0.0, 0.0)))
        net
    }

    def eval(image: Seq[Seq[Double]]) = {
      image.zip(nets).map { s =>
        val (xs, net) = s
        net.evaluate(xs)
      } reduce (_ + _) map { end => end / nets.size }
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

    [scala-execution-context-global-63] INFO neuroflow.nets.DefaultNetwork - [28.07.2017 12:42:03:314] Taking step 999 - Mean Error 0,0294242 - Error Vector 0.03261444866124473  0.026382058841120708  0.02927599065790197
    [scala-execution-context-global-63] INFO neuroflow.nets.DefaultNetwork - [28.07.2017 12:42:03:316] Took 1000 iterations of 1000 with Mean Error = 0,0294098
    Plus classified: Vector(0.8575534162481321, 0.09582553167510581, 0.06881838268270643)
    Plus Rotated classified: Vector(0.8947821350050726, 0.07513311797101342, 0.09147777427535529)
    Heart classified: Vector(0.13802425024512344, 0.8437069779266096, 0.11353161811927058)
    Heart distorted classified: Vector(0.22380692463363228, 0.7291546099037295, 0.10834893153783258)
    Heart rotated classified: Vector(0.18782520100911257, 0.6960125283436842, 0.15336960656603785)
    Random classified: Vector(0.12158710274155932, 0.11611218088283029, 0.8196337328573723)
    [success] Total time: 11 s, completed 28.07.2017 12:42:03

 */