package neuroflow.playground


import java.io.File

import breeze.linalg.max
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Convolution.IntTupler
import neuroflow.core._
import neuroflow.nets.ConvNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    implicit val wp = neuroflow.core.CNN.WeightProvider(-0.0005, 0.001)

    val path = "/Users/felix/github/unversioned/cifar"
    val wps  = "/Users/felix/github/unversioned/cifarWP.nf"

    val classes =
      Seq("airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck")

    val classVecs = classes.zipWithIndex.map { case (c, i) => c -> ~>(ζ(classes.size)).io(_.update(i, 1.0)).t }.toMap

    val train = new File(path + "/train").list().take(50).map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/train/" + s, None) -> classVecs(c)
    }

    val test = new File(path + "/test").list().take(0).map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/test/" + s, None) -> classVecs(c)
    }

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val a = Convolution((32, 32, 3), field = 3`²`, filters = 32, stride = 1, f)
    val b = Convolution( a.dimOut,   field = 4`²`, filters = 32, stride = 1, f)
    val c = Convolution( b.dimOut,   field = 4`²`, filters = 8, stride = 1, f)

    val convs = a :: b :: c :: HNil
    val fully = Output(classes.size, f) :: HNil

    val net = Network(convs ::: fully,
      Settings(
        prettyPrint = true,
        learningRate = {
          case (iter, _) if iter  < 200       => 1E-4
          case (iter, _) if iter  < 750       => 1E-5
          case (iter, _) if iter >= 750       => 1E-6
        },
        updateRule = Momentum(μ = 0.8),
        iterations = 4000,
        parallelism = 32,
        waypoint = Some(Waypoint(nth = 100, ws => IO.File.write(ws, wps)))
      )
    )

    net.train(train.map(_._1), train.map(_._2))

    (train ++ test).foreach {
      case (x, y) =>
        val v = net(x)
        val c = v.data.indexOf(max(v))
        val t = y.data.indexOf(max(y))
        println(s"${classes(t)} classified as ${classes(c)}")
        println(net(x))
    }

    println(net)

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

  }

}

/*


      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:32:475] Iteration 3995 - Mean Error 0,0898069 - Error Vector 0.3813480756077933  1.0719583534368027E-4  ... (10 total)
      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:34:189] Iteration 3996 - Mean Error 0,0897723 - Error Vector 0.3812364693727483  1.0712663542404684E-4  ... (10 total)
      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:35:637] Iteration 3997 - Mean Error 0,0897377 - Error Vector 0.38112488683494417  1.0705741567742608E-4  ... (10 total)
      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:37:268] Iteration 3998 - Mean Error 0,0897032 - Error Vector 0.38101332799190635  1.0698817625021387E-4  ... (10 total)
      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:38:770] Iteration 3999 - Mean Error 0,0896686 - Error Vector 0.3809017928405412  1.0691891702829551E-4  ... (10 total)
      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:40:451] Iteration 4000 - Mean Error 0,0896341 - Error Vector 0.3807902813614159  1.0684963826998159E-4  0.002448611189022405  ... (10 total)
      [run-main-0] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:40:451] Took 4000 iterations of 4000 with Mean Error = 0,0896341
      [scala-execution-context-global-99] INFO neuroflow.nets.ConvNetwork - [14.09.2017 21:32:40:451] Waypoint ...
      frog classified as frog
      DenseVector(0.0, 0.0, 0.0, 0.010115774984676222, 0.0, 0.0, 0.9498347117451181, 0.0, 0.0, 0.0)
      automobile classified as automobile
      DenseVector(0.0, 1.001984828903064, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.11535387582458433, 0.0, 0.0, 0.00345771272257858, 0.0, 0.0, 0.9836664197348073, 0.0010833098700233981, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.20763277293952737, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0465077808924512, 0.0, 0.018586634743406643, 0.0)
      ship classified as ship
      DenseVector(0.0, 0.0, 0.0, 0.08199971537790962, 0.0, 0.0, 0.16445633140418933, 0.0, 0.9746169986438671, 0.0)
      ship classified as ship
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025489920921537998, 6.166237257496446E-4, 0.9959065265411299, 0.0)
      cat classified as cat
      DenseVector(0.029542690862069974, 0.0, 0.0, 0.887927733409417, 0.0, 0.0, 0.2169338135377238, 0.0, 0.0, 0.0)
      deer classified as deer
      DenseVector(0.0, 0.0, 0.04804087630319025, 0.0, 1.027384721948583, 0.0, 0.0, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.05564857731354661, 0.0, 0.010825548268065078, 0.08649947657427046, 0.0, 0.0, 0.9760598058397699, 0.0, 0.0, 0.0)
      airplane classified as airplane
      DenseVector(0.7856386018827942, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.10315948832749296, 0.0, 0.0, 0.0, 0.0, 0.0, 0.589012341458291, 0.0, 0.0, 0.0)
      truck classified as truck
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06403984858037512, 0.0, 0.0, 0.9977896660920145)
      airplane classified as airplane
      DenseVector(1.1693207814555764, 0.0, 0.0, 0.11082329624326899, 0.0, 0.0, 0.0015279002902451105, 0.0, 0.0, 0.0)
      cat classified as cat
      DenseVector(0.004950388621693473, 0.0, 0.0, 0.9432151596569405, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.0015586147185792237, 0.0, 0.0, 0.0, 0.0, 0.0, 1.121203307704296, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8499488903165738, 0.0, 0.0, 0.0)
      dog classified as dog
      DenseVector(0.08695192415708101, 0.0, 0.0, 0.0, 0.0, 1.0009407317408652, 0.008280162217877063, 0.0076732989552991005, 0.0, 0.0)
      deer classified as deer
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.7919779860952899, 0.0, 0.1745891109742435, 0.0, 0.0, 0.0)
      ship classified as ship
      DenseVector(0.10146125409068629, 0.0, 0.0, 0.06577105614573345, 0.02511879610308614, 0.0, 0.0, 0.0, 0.9804848713501978, 0.0)
      cat classified as cat
      DenseVector(0.0, 0.0, 0.0, 1.0773615895502096, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      bird classified as bird
      DenseVector(0.24148328748699988, 0.0, 0.996004843684201, 0.008179390696072531, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7582999578571178, 0.0, 0.015345134192605575, 0.0)
      deer classified as deer
      DenseVector(0.037688535095635314, 0.0, 0.0, 0.0, 0.9917856139827941, 0.0, 0.0, 0.0, 0.0, 0.0)
      airplane classified as airplane
      DenseVector(0.5934742725010158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      cat classified as cat
      DenseVector(0.09597382602654422, 0.0, 0.0, 0.9917048025343932, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      automobile classified as automobile
      DenseVector(0.0, 0.9964135482181058, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      deer classified as deer
      DenseVector(0.0, 0.0, 0.00611707547685896, 0.0, 0.8903627227278044, 0.0, 0.05402643821983042, 0.004159286264382248, 0.010722665453268863, 0.0)
      airplane classified as airplane
      DenseVector(0.5131249580035503, 0.0, 0.0, 0.11608684878951689, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.0, 0.0, 0.016253253231193327, 0.0664352181792329, 0.14574280367024964, 0.0, 0.8139276509377323, 0.0, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0393363690168707, 0.0, 0.0, 0.0)
      bird classified as bird
      DenseVector(0.0, 0.0, 0.9539930244389861, 0.0, 0.014631795211272319, 0.0, 0.0, 0.0, 0.0, 0.0)
      horse classified as horse
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9880753287508621, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.02805438141345859, 0.0, 1.0330335310898437, 0.0, 0.0, 0.0)
      cat classified as cat
      DenseVector(0.0, 0.0, 0.0, 0.9546658512186249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      truck classified as truck
      DenseVector(0.08593146293324767, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025335682951474377, 1.004559323892247)
      airplane classified as airplane
      DenseVector(0.9251177857617044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02961684885636066, 0.0)
      deer classified as deer
      DenseVector(0.0, 0.0, 0.0, 0.0, 1.002835088010524, 0.0, 0.0, 0.0, 0.0, 0.0)
      dog classified as dog
      DenseVector(0.07428720229454533, 0.0, 0.0, 0.07508684791571801, 0.0, 0.9997357017358697, 0.0, 0.0, 0.0, 0.0)
      horse classified as horse
      DenseVector(0.09836107244561541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9761758048047061, 0.005313219067889158, 0.0)
      automobile classified as automobile
      DenseVector(0.1750890393966478, 1.0133412344585855, 0.0, 0.0, 0.0, 0.0, 0.2174092598469606, 0.020976200712400844, 0.0, 0.0)
      frog classified as frog
      DenseVector(0.10001165331215127, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0889241343280542, 0.0, 0.0, 0.0)
      horse classified as horse
      DenseVector(0.0, 0.0, 0.0032469062550090433, 0.08407307491882783, 0.0, 0.0, 0.0, 0.9929474536146318, 0.0, 0.0)
      truck classified as truck
      DenseVector(0.06825251920111448, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999327213430057)
      automobile classified as automobile
      DenseVector(0.01632886132502076, 0.9956675685489618, 0.0, 0.03613803028345266, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      bird classified as bird
      DenseVector(0.06200651098015267, 0.0, 1.003919870633878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      horse classified as horse
      DenseVector(0.0551081589849295, 0.0, 1.765334504467684E-4, 0.0, 0.0, 0.0, 0.0, 1.0125372139795619, 0.0, 0.0)
      horse classified as horse
      DenseVector(0.0, 0.0, 0.0, 0.014084794590542637, 0.0, 0.0, 0.0, 1.005992176007889, 0.0, 0.0)
      ship classified as ship
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.03872877775070167, 0.0, 0.05136490201261748, 0.0, 0.9680328663814695, 0.0)
      airplane classified as airplane
      DenseVector(0.7857113370345473, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      cat classified as cat
      DenseVector(0.0, 0.0, 0.0, 0.387796022826571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

 */