package neuroflow.playground

import neuroflow.application.classification.Image._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.WeightProvider.randomWeights
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {
    // Training
    val plus = extractRgb(getFile("img/plus.png")).grouped(30).toList
    val heart = extractRgb(getFile("img/heart.png")).grouped(30).toList

    // Testing
    val heartDistorted = extractRgb(getFile("img/heart_distorted.png")).grouped(30).toList
    val heartRotated = extractRgb(getFile("img/heart_rotated.png")).grouped(30).toList
    val plusRotated = extractRgb(getFile("img/plus_rotated.png")).grouped(30).toList
    val random = extractRgb(getFile("img/random.png")).grouped(30).toList

    println(s"Training ${plus.size + heart.size} samples...")

    val training = plus.zip(heart).zip(random)
    val nets = training.par.map { sample =>
      val (ph, r) = sample
      val (p, h) = ph
      val net = Network(Input(p.size) :: Hidden(20, Sigmoid.apply) :: Output(3, Sigmoid.apply) :: Nil, Settings(numericGradient = true, Δ = 0.00001, verbose = true))
      net.train(Seq(p, h, r), Seq(Seq(1.0, 0.0, 0.0), Seq(0.0, 1.0, 0.0), Seq(0.0, 0.0, 1.0)), 20.0, 0.001, 1000)
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

[INFO] [12.01.2016 16:23:26:323] [ForkJoinPool-1-worker-1] Took 52 iterations of 1000 with error 6.472253253659001E-4  9.858371755480464E-4  2.2336574126036537E-4
Plus classified: List(0.9735662669022744, 0.024478967499628992, 0.009637646096690936)
Plus Rotated classified: List(0.284005243372372, 0.6192290192921474, 0.1413368098813749)
Heart classified: List(0.02555783474629203, 0.9698023355180021, 0.021333456200719088)
Heart distorted classified: List(0.07276547774260399, 0.8130479912043663, 0.05636021910245942)
Heart rotated classified: List(0.1900615743518672, 0.6629531269148059, 0.09065040782648792)
Random classified: List(0.00786187832984005, 0.01949796312984, 0.9768238604402418)

[success] Total time: 13 s, completed 12.01.2016 16:23:26

 */
