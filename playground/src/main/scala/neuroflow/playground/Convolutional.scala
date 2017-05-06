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
    val xsys = raw ++ (1 to 33).flatMap { _ =>
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
        Settings(iterations = 200, precision = 1E-3, specifics = Some(Map("m" -> 7)))
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

             Identifier: Lya
             Network: neuroflow.nets.ConvolutionalNetwork
             Layout: [30000 In, 180 Cn(x), 23 Cn(x), 6 Cn(x), 6 H(σ), 3 Out(σ)]
             Number of Weights: 263




    Mai 06, 2017 12:20:40 PM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader2827254945879314551netlib-native_system-osx-x86_64.jnilib
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:21:30:718] Taking step 0. Step Size: 0,1683. Val and Grad Norm: 22,6727 (rel: 0,108) 0,284677
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:21:47:628] Taking step 1. Step Size: 1,000. Val and Grad Norm: 22,6629 (rel: 0,000435) 0,204912
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:22:38:338] Taking step 2. Step Size: 2,250. Val and Grad Norm: 22,6155 (rel: 0,00209) 0,935479
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 25.576753096253366 rhs: 22.615426291998347 cdd: 47.881272415674815
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.010000000000000002 fval: 22.587140465621925 rhs: 22.615456312596177 cdd: -1.621756150347251
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:23:29:329] Taking step 3. Step Size: 0,01000. Val and Grad Norm: 22,5871 (rel: 0,00125) 1,55407
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.34309620551693887 fval: 22.574585023174397 rhs: 22.58713839173258 cdd: -0.0018767797055608444
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:24:03:557] Taking step 4. Step Size: 0,3431. Val and Grad Norm: 22,5746 (rel: 0,000556) 1,89218
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:24:20:749] Taking step 5. Step Size: 1,000. Val and Grad Norm: 22,5304 (rel: 0,00196) 2,00302
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.5131106522853675 fval: 22.79557793431563 rhs: 22.53033812064413 cdd: 0.7250335876144767
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.11800861922449768 fval: 22.416160005682656 rhs: 22.53039284571433 cdd: -0.34037514701495536
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:25:12:647] Taking step 6. Step Size: 0,1180. Val and Grad Norm: 22,4162 (rel: 0,00507) 2,30026
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:25:29:955] Taking step 7. Step Size: 1,000. Val and Grad Norm: 22,2100 (rel: 0,00920) 2,01631
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:25:47:506] Taking step 8. Step Size: 1,000. Val and Grad Norm: 21,8946 (rel: 0,0142) 7,28550
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 21.774778405543525 rhs: 21.894552087492674 cdd: -1.2526472478223503
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.19 fval: 21.663471248855807 rhs: 21.89454199905702 cdd: -1.174704488069071
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.271 fval: 21.581686096407125 rhs: 21.89453291946493 cdd: -0.7746952071104065
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:27:06:803] Taking step 9. Step Size: 0,2710. Val and Grad Norm: 21,5817 (rel: 0,0143) 10,0603
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 24.8280134236792 rhs: 21.581614605005303 cdd: 38.162769553749115
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.010000000000000002 fval: 21.522414231473796 rhs: 21.581678947266944 cdd: -4.328898127425198
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:28:04:313] Taking step 10. Step Size: 0,01000. Val and Grad Norm: 21,5224 (rel: 0,00275) 12,1169
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:28:22:953] Taking step 11. Step Size: 1,000. Val and Grad Norm: 20,7562 (rel: 0,0356) 24,8932
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.44838165085060566 fval: 19.478498823251478 rhs: 20.756030822228677 cdd: -0.349330974629883
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:29:02:178] Taking step 12. Step Size: 0,4484. Val and Grad Norm: 19,4785 (rel: 0,0616) 12,1657
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.5023295973411603 fval: 18.986916266999227 rhs: 19.478438147326465 cdd: -0.4015094631960675
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:29:41:571] Taking step 13. Step Size: 0,5023. Val and Grad Norm: 18,9869 (rel: 0,0252) 16,8103
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:29:59:504] Taking step 14. Step Size: 1,000. Val and Grad Norm: 18,6663 (rel: 0,0169) 59,5968
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:30:18:810] Taking step 15. Step Size: 1,000. Val and Grad Norm: 17,7972 (rel: 0,0466) 59,9117
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:31:13:579] Taking step 16. Step Size: 2,250. Val and Grad Norm: 15,5072 (rel: 0,129) 93,6603
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.29537337722062573 fval: 38.20516206513837 rhs: 15.506239577409424 cdd: 34.43425811297212
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.029537337722062573 fval: 15.106632893964894 rhs: 15.507115057747356 cdd: 7.175009864343856
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:32:07:843] Taking step 17. Step Size: 0,02954. Val and Grad Norm: 15,1066 (rel: 0,0258) 220,273
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:32:25:926] Taking step 18. Step Size: 1,000. Val and Grad Norm: 14,3751 (rel: 0,0484) 37,9650
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:32:44:075] Taking step 19. Step Size: 1,000. Val and Grad Norm: 13,0148 (rel: 0,0946) 89,1051
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.3043609537683546 fval: 11.047656554560923 rhs: 13.014306608806349 cdd: -1.4520683706712867
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:33:19:788] Taking step 20. Step Size: 0,3044. Val and Grad Norm: 11,0477 (rel: 0,151) 48,5473
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:33:37:529] Taking step 21. Step Size: 1,000. Val and Grad Norm: 10,6987 (rel: 0,0316) 18,7468
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.28063359096252827 fval: 9.900404001073657 rhs: 10.698611753060234 cdd: -2.118992385429273
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:34:13:288] Taking step 22. Step Size: 0,2806. Val and Grad Norm: 9,90040 (rel: 0,0746) 131,133
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.8508828713357097 fval: 6.850037685394911 rhs: 9.900255849039388 cdd: 1.9141080346224726
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.7657945842021388 fval: 6.731148155616567 rhs: 9.900270664242814 cdd: 0.9873853037333049
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:35:07:499] Taking step 23. Step Size: 0,7658. Val and Grad Norm: 6,73115 (rel: 0,320) 1400,10
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.21062213416085984 fval: 5.159103600464659 rhs: 6.730919646968542 cdd: -5.168774584953269
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:35:43:899] Taking step 24. Step Size: 0,2106. Val and Grad Norm: 5,15910 (rel: 0,234) 1164,63
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.5971926235597806 fval: 4.5130874897556446 rhs: 5.158963855440344 cdd: 1.9068363699645081
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:36:19:466] Taking step 25. Step Size: 0,5972. Val and Grad Norm: 4,51309 (rel: 0,125) 305,533
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.5805240548437642 fval: 3.0261246494131977 rhs: 4.512227669515861 cdd: -0.6933418018702692
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:36:55:221] Taking step 26. Step Size: 0,5805. Val and Grad Norm: 3,02612 (rel: 0,329) 2,73112
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:37:13:218] Taking step 27. Step Size: 1,000. Val and Grad Norm: 2,91215 (rel: 0,0377) 2,89303
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.11729585122298225 fval: 5.916230328554435 rhs: 2.9119354410259883 cdd: 389.93853401626217
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07116303987453923 fval: 2.455527584622323 rhs: 2.912019749734561 cdd: -10.203363530482676
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:38:07:633] Taking step 28. Step Size: 0,07116. Val and Grad Norm: 2,45553 (rel: 0,157) 3,79566
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 1.7613819716662937 rhs: 2.455514438108678 cdd: -1.2065422083524064
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.19 fval: 1.6563549629283045 rhs: 2.4555026062463976 cdd: -1.1281527490633585
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:39:01:784] Taking step 29. Step Size: 0,1900. Val and Grad Norm: 1,65635 (rel: 0,325) 1,26860
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 1.3934059505330554 rhs: 1.656326330708362 cdd: -2.409698324751718
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:39:37:332] Taking step 30. Step Size: 0,1000. Val and Grad Norm: 1,39341 (rel: 0,159) 1,11010
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 9.364013041389187 rhs: 1.3933904088988796 cdd: 27.836687413746542
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.010000000000000002 fval: 1.3780211098279107 rhs: 1.3934043963696379 cdd: -1.5228843427494652
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.019000000000000003 fval: 1.3644409820450973 rhs: 1.393402997622562 cdd: -1.4949915790088733
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.027100000000000006 fval: 1.3524326230831336 rhs: 1.3934017387501938 cdd: -1.4700970653006717
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.034390000000000004 fval: 1.341796867484221 rhs: 1.3934006057650623 cdd: -1.4478605402424447
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.040951 fval: 1.5883606958695493 rhs: 1.393399586078444 cdd: 143.32286507274398
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.036363036058899706 fval: 1.3389461617427938 rhs: 1.3934002991230159 cdd: -1.4417683799914518
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.03682183245300973 fval: 1.3382850547060456 rhs: 1.3934002278185587 cdd: -1.4401186652572833
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.03723474920770876 fval: 1.3376907936646427 rhs: 1.3934001636445472 cdd: -1.4380987826109084
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.03760637428693788 fval: 1.3371568779883607 rhs: 1.393400105887937 cdd: -1.434909114075704
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:42:55:249] Failure! Resetting history: breeze.optimize.FirstOrderException: Line search zoom failed
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.4381423991536485 fval: 1.1419365173426124 rhs: 1.3933519568704262 cdd: 0.005652712089239197
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:43:49:553] Taking step 31. Step Size: 0,4381. Val and Grad Norm: 1,14194 (rel: 0,180) 0,955506
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:44:07:802] Taking step 32. Step Size: 1,000. Val and Grad Norm: 0,944127 (rel: 0,173) 0,656794
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:44:26:326] Taking step 33. Step Size: 1,000. Val and Grad Norm: 0,451308 (rel: 0,522) 0,287120
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:44:43:991] Taking step 34. Step Size: 1,000. Val and Grad Norm: 0,219436 (rel: 0,514) 0,144822
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:45:01:683] Taking step 35. Step Size: 1,000. Val and Grad Norm: 0,111122 (rel: 0,494) 0,0688554
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:45:19:464] Taking step 36. Step Size: 1,000. Val and Grad Norm: 0,0556623 (rel: 0,499) 0,0381904
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:45:37:387] Taking step 37. Step Size: 1,000. Val and Grad Norm: 0,0341472 (rel: 0,387) 0,0632072
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:45:55:281] Taking step 38. Step Size: 1,000. Val and Grad Norm: 0,0124108 (rel: 0,637) 0,0127771
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:46:13:306] Taking step 39. Step Size: 1,000. Val and Grad Norm: 0,00779591 (rel: 0,372) 0,00733234
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:46:31:427] Taking step 40. Step Size: 1,000. Val and Grad Norm: 0,00359202 (rel: 0,539) 0,00301911
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:46:49:720] Taking step 41. Step Size: 1,000. Val and Grad Norm: 0,00186199 (rel: 0,482) 0,00144738
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 12:47:07:987] Taking step 42. Step Size: 1,000. Val and Grad Norm: 0,000920937 (rel: 0,505) 0,000683625
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Converged because Error function is sufficiently minimal.
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.999999999881884, 4.9744811322734E-5, 0.0015415495406166917)
    Vector(8.627570924429739E-5, 0.9964572719878335, 0.004712591014250767)
    Vector(0.0013509938910771509, 0.004402188206827236, 0.9952136713539983)
    Vector(0.9999733684834847, 4.5369052732505984E-4, 0.026082814255508454)
    Vector(0.9999989172611808, 9.656590544378496E-6, 0.8083714963673374)
    [success] Total time: 1600 s, completed 06.05.2017 12:47:08

 */
