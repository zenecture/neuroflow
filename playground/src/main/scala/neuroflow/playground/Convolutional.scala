package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image.extractRgb
import neuroflow.application.processor.Util.getResourceFile
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.ConvolutionalNetwork._
import shapeless._

import scala.util.Random

/**
  * @author bogdanski
  * @since 03.05.17
  */
object Convolutional {

  def apply = {

    implicit val wp = CNN.WeightProvider(-0.2, 0.2)

    val f = Sigmoid
    val g = Linear

    val raw = ->(
      (extractRgb(getResourceFile("img/larger/heart.png")),           ->(1.0, 0.0, 0.0)),
      (extractRgb(getResourceFile("img/larger/plus.png")),            ->(0.0, 1.0, 0.0)),
      (extractRgb(getResourceFile("img/larger/random.png")),          ->(0.0, 0.0, 1.0))
    )

    // Generality peek.
    val test = ->(
      extractRgb(getResourceFile("img/larger/heart_distorted.png")),
      extractRgb(getResourceFile("img/larger/heart_rotated.png"))
    )

    def noise = if (Random.nextDouble > 0.5) 0.01 else -0.01

    // This corresponds to 3 + (3 * 33) = 102 pictures à 100px * 100px * rgb
    val xsys = raw ++ (1 to 100).flatMap { _ =>
      raw.map(l => l._1.map(_ + noise) -> l._2)
    }

    val dim = xsys.head._1.size

    val net =
      Network(
        Input(dim)              ::
        LinConvolution(
          filters   = 1,
          fieldSize = 180,
          stride    = 166,
          padding   = 0, g)     ::
        LinConvolution(
          filters   = 1,
          fieldSize = 23,
          stride    = 7,
          padding   = 0, g)     ::
        LinConvolution(
            filters   = 1,
            fieldSize = 6,
            stride    = 3,
            padding   = 0, g)   ::
        Hidden(6, f)            ::
        Output(3, f)            :: HNil,
        Settings(iterations = 200, specifics = Some(Map("m" -> 7)))
      )

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = (xsys.map(_._1) ++ test).map(x => net.evaluate(x))

    res.foreach(println)

  }

}

/*





                 _   __                      ________
                / | / /__  __  ___________  / ____/ /___ _      __
               /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
              / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
             /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


             Version 0.700

             Identifier: JQt
             Network: neuroflow.nets.ConvolutionalNetwork
             Layout: [30000 I, 180 Cn(x), 23 Cn(x), 6 Cn(x), 3 O(σ)]
             Number of Weights: 227




    Mai 05, 2017 9:24:41 PM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader4306766430465248085netlib-native_system-osx-x86_64.jnilib
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.04350441699026276 fval: 72.66377033328446 rhs: 76.14420547398201 cdd: -111.35157567632842
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:27:46:716] Taking step 0. Step Size: 0,04350. Val and Grad Norm: 72,6638 (rel: 0,0457) 24,7098
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.11835690679328514 fval: 71.96517038972756 rhs: 72.66340103450888 cdd: 30.030041502426165
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07466939490237916 fval: 71.26934182217934 rhs: 72.66353734885746 cdd: 1.5727392495326669
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:30:03:368] Taking step 1. Step Size: 0,07467. Val and Grad Norm: 71,2693 (rel: 0,0192) 61,5160
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:30:50:232] Taking step 2. Step Size: 1,000. Val and Grad Norm: 69,7667 (rel: 0,0211) 105,366
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:31:43:200] Taking step 3. Step Size: 1,000. Val and Grad Norm: 65,3206 (rel: 0,0637) 23,8442
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:32:33:099] Taking step 4. Step Size: 1,000. Val and Grad Norm: 64,0574 (rel: 0,0193) 48,2556
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:33:25:941] Taking step 5. Step Size: 1,000. Val and Grad Norm: 62,1567 (rel: 0,0297) 56,9316
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:34:16:042] Taking step 6. Step Size: 1,000. Val and Grad Norm: 50,5821 (rel: 0,186) 161,087
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 50.303091012040184 rhs: 50.58206902348485 cdd: 3.3990051617384607
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:35:47:311] Taking step 7. Step Size: 0,1000. Val and Grad Norm: 50,3031 (rel: 0,00552) 151,737
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.3965381131085408 fval: 46.01630579505958 rhs: 50.30226218917457 cdd: -0.4677141921406745
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:37:20:233] Taking step 8. Step Size: 0,3965. Val and Grad Norm: 46,0163 (rel: 0,0852) 130,285
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:38:16:586] Taking step 9. Step Size: 1,000. Val and Grad Norm: 28,2207 (rel: 0,387) 206,219
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.4989182154690902 fval: 17.367214823472608 rhs: 28.219300636120245 cdd: -8.04785784022798
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:39:57:525] Taking step 10. Step Size: 0,4989. Val and Grad Norm: 17,3672 (rel: 0,385) 432,086
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.42718095734559725 fval: 11.097909312101429 rhs: 17.36620976559572 cdd: -2.0888531708417597
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:41:39:280] Taking step 11. Step Size: 0,4272. Val and Grad Norm: 11,0979 (rel: 0,361) 703,160
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:42:27:359] Taking step 12. Step Size: 1,000. Val and Grad Norm: 0,0960886 (rel: 0,991) 12,6335
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:43:16:304] Taking step 13. Step Size: 1,000. Val and Grad Norm: 0,0697763 (rel: 0,274) 8,03039
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:44:05:764] Taking step 14. Step Size: 1,000. Val and Grad Norm: 0,0271925 (rel: 0,610) 1,74657
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:44:58:663] Taking step 15. Step Size: 1,000. Val and Grad Norm: 0,0116488 (rel: 0,572) 1,73704
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.010867273687803377 rhs: 0.011647775175121208 cdd: 0.049963115973730994
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:46:37:553] Taking step 16. Step Size: 0,1000. Val and Grad Norm: 0,0108673 (rel: 0,0671) 1,50844
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:47:25:497] Taking step 17. Step Size: 1,000. Val and Grad Norm: 0,00395729 (rel: 0,636) 0,370762
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:48:14:680] Taking step 18. Step Size: 1,000. Val and Grad Norm: 0,00216343 (rel: 0,453) 0,161592
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:49:05:581] Taking step 19. Step Size: 1,000. Val and Grad Norm: 0,000978037 (rel: 0,548) 0,0524090
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:49:57:510] Taking step 20. Step Size: 1,000. Val and Grad Norm: 0,000468419 (rel: 0,521) 0,0172437
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:50:48:012] Taking step 21. Step Size: 1,000. Val and Grad Norm: 0,000218885 (rel: 0,533) 0,00875122
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:51:39:419] Taking step 22. Step Size: 1,000. Val and Grad Norm: 0,000121107 (rel: 0,447) 0,0149780
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:52:30:600] Taking step 23. Step Size: 1,000. Val and Grad Norm: 3,20293e-05 (rel: 0,736) 0,00324349
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:53:22:144] Taking step 24. Step Size: 1,000. Val and Grad Norm: 1,93308e-05 (rel: 0,167) 0,00196770
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [05.05.2017 21:54:10:270] Taking step 25. Step Size: 1,000. Val and Grad Norm: 8,51578e-06 (rel: 0,142) 0,000893654
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Converged because Error function is sufficiently minimal.
    Vector(0.9999170946150057, 4.7138722962651147E-4, 1.0286894308957172E-7)
    Vector(1.9122695507201226E-6, 0.9999999999999987, 2.3288711172022705E-23)
    Vector(3.915818546548453E-8, 2.110060354797627E-15, 0.9999999999999989)
    Vector(0.9999410614159226, 4.567477478717262E-4, 9.707765396462971E-8)
    Vector(1.5997104101714039E-6, 0.9999999999999982, 2.8541862081048865E-23)
    Vector(3.1246733017623996E-8, 2.3917013985706463E-15, 0.9999999999999987)
    Vector(0.9999356631735693, 4.4922869546237527E-4, 1.0850318766833171E-7)
    Vector(2.6569360147502233E-6, 0.9999999999999982, 2.5511407117425855E-23)
    Vector(3.6672540622978524E-8, 2.1930973691181105E-15, 0.9999999999999991)
    Vector(0.9998775658809069, 4.7929558059745664E-4, 8.161522718730595E-8)
    Vector(3.4596406979752513E-6, 0.9999999999999989, 1.761453755333154E-23)
    Vector(5.721292497086382E-8, 1.845184041348292E-15, 0.9999999999999991)
    Vector(0.9999169619058667, 5.114750905634736E-4, 9.339959418079172E-8)
    Vector(1.7905438526973204E-6, 0.9999999999999984, 2.1458481725153456E-23)
    Vector(4.0729310333047717E-8, 2.2075793623789303E-15, 0.9999999999999989)
    Vector(0.9999135217250017, 3.530313703842876E-4, 1.19731115508602E-7)
    ...
    Vector(0.9998643452912241, 4.286166864132157E-4, 1.2166666147021908E-7)
    Vector(2.1232142518556146E-6, 0.9999999999999982, 2.754214654554028E-23)
    Vector(3.0053335372015774E-8, 2.305608387298825E-15, 0.9999999999999989)
    Vector(0.999999903658487, 0.9639988544151356, 1.135114630981481E-8)
    Vector(0.0714594136245913, 0.9999979049160299, 5.410785189406003E-12)
    [success] Total time: 1778 s, completed 05.05.2017 21:54:11



 */