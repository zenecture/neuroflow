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

             Identifier: tYY
             Network: neuroflow.nets.ConvolutionalNetwork
             Layout: [30000 In, 180 Cn(x), 23 Cn(x), 6 Cn(x), 6 H(σ), 3 Out(σ)]
             Number of Weights: 263




    Mai 06, 2017 12:17:43 AM com.github.fommil.jni.JniLoader liberalLoad
    INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader5040029453944573562netlib-native_system-osx-x86_64.jnilib
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:20:32:966] Taking step 0. Step Size: 0,06010. Val and Grad Norm: 67,4281 (rel: 0,0982) 1,26091
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:21:39:734] Taking step 1. Step Size: 1,000. Val and Grad Norm: 67,3701 (rel: 0,000860) 0,538112
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:22:38:434] Taking step 2. Step Size: 1,000. Val and Grad Norm: 67,3466 (rel: 0,000349) 0,464489
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:23:46:827] Taking step 3. Step Size: 1,000. Val and Grad Norm: 67,3113 (rel: 0,000524) 0,677846
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:24:45:142] Taking step 4. Step Size: 1,000. Val and Grad Norm: 67,2479 (rel: 0,000942) 1,26886
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:27:34:089] Taking step 5. Step Size: 2,250. Val and Grad Norm: 67,0248 (rel: 0,00332) 3,37382
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 93.7616003303296 rhs: 67.0245734639925 cdd: 282.15977106157476
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.010000000000000002 fval: 67.0746023103864 rhs: 67.02479402792679 cdd: 58.18867855599132
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.005114100289218559 fval: 66.94989632613016 rhs: 67.02480600185194 cdd: -0.02855312087146089
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:31:21:874] Taking step 6. Step Size: 0,005114. Val and Grad Norm: 66,9499 (rel: 0,00112) 5,98559
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:37:05:865] Taking step 7. Step Size: 7,594. Val and Grad Norm: 65,0024 (rel: 0,0291) 73,0031
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 64.70480141425276 rhs: 65.00208588311092 cdd: -4.258961999855629
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:38:58:588] Taking step 8. Step Size: 0,1000. Val and Grad Norm: 64,7048 (rel: 0,00458) 253,080
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.16969165804144049 fval: 51.88400808528925 rhs: 64.70378566269063 cdd: -16.4532195372268
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:40:57:653] Taking step 9. Step Size: 0,1697. Val and Grad Norm: 51,8840 (rel: 0,198) 569,758
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.11400471369443821 fval: 50.23897213247702 rhs: 51.88360088826348 cdd: -26.74123304437194
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:42:46:234] Taking step 10. Step Size: 0,1140. Val and Grad Norm: 50,2390 (rel: 0,0317) 793,528
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.13181250625577257 fval: 53.49231599933833 rhs: 50.23824796106479 cdd: 108.16176486985876
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.046537141679564756 fval: 46.90963381858841 rhs: 50.23871645971449 cdd: -27.709079486574637
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:45:25:174] Taking step 11. Step Size: 0,04654. Val and Grad Norm: 46,9096 (rel: 0,0663) 838,688
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:46:18:652] Taking step 12. Step Size: 1,000. Val and Grad Norm: 44,1830 (rel: 0,0581) 713,259
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:47:11:612] Taking step 13. Step Size: 1,000. Val and Grad Norm: 31,7538 (rel: 0,281) 97,2414
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:48:07:500] Taking step 14. Step Size: 1,000. Val and Grad Norm: 31,1952 (rel: 0,0176) 28,9443
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:49:10:875] Taking step 15. Step Size: 1,000. Val and Grad Norm: 31,1048 (rel: 0,00290) 24,2229
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:50:16:124] Taking step 16. Step Size: 1,000. Val and Grad Norm: 30,3754 (rel: 0,0234) 14,3248
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.5596087368539162 fval: 29.335670919763583 rhs: 30.375298413956635 cdd: -1.3937237704988992
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:52:35:303] Taking step 17. Step Size: 0,5596. Val and Grad Norm: 29,3357 (rel: 0,0342) 21,3486
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 1.3654377227368744 fval: 19.565920465893296 rhs: 29.33453116265287 cdd: -0.8295842578420098
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:55:47:428] Taking step 18. Step Size: 1,365. Val and Grad Norm: 19,5659 (rel: 0,333) 75,4594
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:56:53:706] Taking step 19. Step Size: 1,000. Val and Grad Norm: 18,0713 (rel: 0,0764) 22,2281
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:57:56:581] Taking step 20. Step Size: 1,000. Val and Grad Norm: 13,0865 (rel: 0,276) 13,1466
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 00:58:59:748] Taking step 21. Step Size: 1,000. Val and Grad Norm: 2,25929 (rel: 0,827) 2,87706
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.6269415684697683 fval: 1.9556090866396278 rhs: 2.259138844613393 cdd: -0.9332570344645443
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:00:56:413] Taking step 22. Step Size: 0,6269. Val and Grad Norm: 1,95561 (rel: 0,134) 1,26009
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.2884742895736645 fval: 3.611219180972382 rhs: 1.9555887678102997 cdd: -0.2103013174600037
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.02884742895736645 fval: 1.9286514796251424 rhs: 1.955607054756695 cdd: -0.6888774224198555
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.054810115018996255 fval: 1.9499199569415113 rhs: 1.9556052260620556 cdd: 55.291294854919116
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.04583508442969883 fval: 1.9170558307764467 rhs: 1.955605858222776 cdd: -0.671199076489968
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.05133951705939024 fval: 1.9140992383603315 rhs: 1.95560547051534 cdd: 0.07838285844545823
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:06:35:189] Taking step 23. Step Size: 0,05134. Val and Grad Norm: 1,91410 (rel: 0,0212) 13,4606
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:07:28:546] Taking step 24. Step Size: 1,000. Val and Grad Norm: 1,70088 (rel: 0,111) 0,922564
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.6527371520576315 fval: 1.423409941347949 rhs: 1.7008342243646883 cdd: 2.7582238160439005
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.46926795453932646 fval: 1.4575541687625289 rhs: 1.70084576462717 cdd: -0.3957602406072655
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.5951072413808897 fval: 1.4116091042814087 rhs: 1.7008378493017622 cdd: -0.26364907353516803
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:11:05:021] Taking step 25. Step Size: 0,5951. Val and Grad Norm: 1,41161 (rel: 0,170) 28,6499
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.2569556011555467 fval: 1.3191283095870827 rhs: 1.4115957574605302 cdd: -0.325858694198835
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:12:52:556] Taking step 26. Step Size: 0,2570. Val and Grad Norm: 1,31913 (rel: 0,0655) 0,525509
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 1.860996495261679 rhs: 1.319124204390792 cdd: 512.0461845727305
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.06593212189758893 fval: 1.2926346325635993 rhs: 1.3191256029440603 cdd: -0.39306844567027366
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.08784766378180446 fval: 1.517752721516682 rhs: 1.3191247032680478 cdd: -0.5170652642796051
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.06812367608601048 fval: 1.2917737767119508 rhs: 1.319125512976459 cdd: -0.392544282863097
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07009607485558988 fval: 1.290999983788125 rhs: 1.3191254320056178 cdd: -0.39207815271147894
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07187123374821135 fval: 1.2903043421866989 rhs: 1.3191253591318608 cdd: -0.3916798094841873
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07346887675157066 fval: 1.2896788034435414 rhs: 1.3191252935454796 cdd: -0.3914308779255284
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07490675545459403 fval: 1.2891158910382714 rhs: 1.3191252345177364 cdd: -0.39166833512607324
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07620084628731508 fval: 1.2886083150726293 rhs: 1.3191251813927676 cdd: -0.3929425423966087
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07736552803676401 fval: 1.288149882229239 rhs: 1.3191251335802956 cdd: -0.39394841211421106
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:23:07:674] Failure! Resetting history: breeze.optimize.FirstOrderException: Line search zoom failed
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:27:50:247] Taking step 27. Step Size: 3,375. Val and Grad Norm: 0,502571 (rel: 0,619) 0,727703
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.4978251132176075 rhs: 0.502560934363909 cdd: -1.0738406490741357
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.19 fval: 0.602931292204204 rhs: 0.5025517273070136 cdd: -0.8107209846539729
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.109 fval: 0.48816755533302836 rhs: 0.5025600136582195 cdd: -1.3374014327428738
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1171 fval: 0.47957645541469807 rhs: 0.5025591850230989 cdd: -1.0910893424344936
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.12439 fval: 0.47170516056922196 rhs: 0.5025584392514904 cdd: -0.9950589325554438
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.130951 fval: 0.46471619718695234 rhs: 0.5025577680570427 cdd: -1.1578965891409823
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1368559 fval: 0.45872848186520887 rhs: 0.5025571639820398 cdd: -1.3001328479038294
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.14217031 fval: 0.45338758048379957 rhs: 0.5025566203145372 cdd: -0.94007517005593
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.146953279 fval: 0.44892707240652885 rhs: 0.5025561310137848 cdd: -0.9252926721183816
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.15125795109999998 fval: 0.44478932963537654 rhs: 0.5025556906431077 cdd: -0.8715656321642506
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:38:42:694] Taking step 28. Step Size: 0,1513. Val and Grad Norm: 0,444789 (rel: 0,115) 0,905533
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.167584498736856 fval: 0.5807705013856459 rhs: 0.4447595225135592 cdd: -0.6816231857209046
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.0260501704307331 fval: 0.40115575383822766 rhs: 0.444784696267677 cdd: -1.4448949365064283
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:41:21:815] Taking step 29. Step Size: 0,02605. Val and Grad Norm: 0,401156 (rel: 0,0981) 0,675630
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.47632921888089375 fval: 2.3073657225724444 rhs: 0.40112743540426576 cdd: 6.921602557070639
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.047632921888089375 fval: 0.508961685113188 rhs: 0.4011529219948315 cdd: -0.5888574559447934
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.004763292188808938 fval: 0.39833353450997777 rhs: 0.401155470653888 cdd: -0.5904804014107641
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.009050255158736982 fval: 0.3955991446400482 rhs: 0.40115521578798236 cdd: -0.5860440196072391
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.012908521831672222 fval: 0.39548024875334853 rhs: 0.4011549864086673 cdd: 7.049714635370956
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.011579751116992491 fval: 0.39418037312688825 rhs: 0.4011550654059327 cdd: -0.45103250602321193
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:47:36:565] Taking step 30. Step Size: 0,01158. Val and Grad Norm: 0,394180 (rel: 0,0174) 2,30466
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:48:29:313] Taking step 31. Step Size: 1,000. Val and Grad Norm: 0,0512627 (rel: 0,870) 0,167254
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:49:23:001] Taking step 32. Step Size: 1,000. Val and Grad Norm: 0,0302156 (rel: 0,411) 0,0869793
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:50:19:138] Taking step 33. Step Size: 1,000. Val and Grad Norm: 0,0179582 (rel: 0,406) 0,0459170
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:51:16:178] Taking step 34. Step Size: 1,000. Val and Grad Norm: 0,00950991 (rel: 0,470) 0,0227347
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1258079061954147 fval: 0.008715251154778942 rhs: 0.009509822318758366 cdd: -0.006042370519593186
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.21322711557587326 fval: 0.008202878670322905 rhs: 0.009509764633340396 cdd: -0.005683588725534925
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:54:02:237] Taking step 35. Step Size: 0,2132. Val and Grad Norm: 0,00820288 (rel: 0,137) 0,0195307
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.007482662278655397 rhs: 0.008202803259494998 cdd: -0.006873665435803714
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.19 fval: 0.0068891367717253595 rhs: 0.008202735389749884 cdd: -0.0063235508841876726
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 01:56:42:493] Taking step 36. Step Size: 0,1900. Val and Grad Norm: 0,00688914 (rel: 0,160) 0,0163573
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.37697920949415675 fval: 0.03746386797274856 rhs: 0.006888901285927528 cdd: -0.030705186119306976
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.03769792094941568 fval: 0.006657663778038363 rhs: 0.006889113223145576 cdd: -0.0060350904464701125
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.07162604980388979 fval: 0.006456048673854467 rhs: 0.006889092029423772 cdd: -0.005850809745703706
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.10216136577291648 fval: 0.006279864205303497 rhs: 0.006889072955074147 cdd: -0.00568976747036338
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.12964315014504052 fval: 0.0061254483440511135 rhs: 0.006889055788159485 cdd: -0.005548619058457483
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:02:01:697] Taking step 37. Step Size: 0,1296. Val and Grad Norm: 0,00612545 (rel: 0,111) 0,0145222
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.005579216144439803 rhs: 0.0061253911092282115 cdd: -0.005209324483026093
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.19 fval: 0.021956680483467373 rhs: 0.006125339597887599 cdd: -0.01951455898351597
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.109 fval: 0.005532530708084751 rhs: 0.00612538595809415 cdd: -0.0051653786492200145
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1171 fval: 0.005490850642373088 rhs: 0.006125381322073495 cdd: -0.005126144589959811
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:06:27:955] Taking step 38. Step Size: 0,1171. Val and Grad Norm: 0,00549085 (rel: 0,104) 0,0130020
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.037835870218070666 rhs: 0.0054907990897564785 cdd: -0.033761379207838324
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.010000000000000002 fval: 0.005439428444824829 rhs: 0.005490845487111427 cdd: -0.005246324302146329
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.019000000000000003 fval: 0.01417750794301222 rhs: 0.005490840847375932 cdd: 0.17930816533259633
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.010900000000000002 fval: 0.005434625537331541 rhs: 0.005490845023137877 cdd: -0.005435755933606224
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.011710000000000002 fval: 0.005430251793861684 rhs: 0.005490844605561683 cdd: -0.0051118556177268465
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.012439000000000002 fval: 0.0054279658031734105 rhs: 0.005490844229743108 cdd: 9.881388694949428E-4
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:18:27:439] Taking step 39. Step Size: 0,01244. Val and Grad Norm: 0,00542797 (rel: 0,0115) 0,0751084
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:19:21:681] Taking step 40. Step Size: 1,000. Val and Grad Norm: 0,00481854 (rel: 0,112) 0,0111011
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.004344077012762036 rhs: 0.004818489614957057 cdd: -0.004452058375219429
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:21:07:914] Taking step 41. Step Size: 0,1000. Val and Grad Norm: 0,00434408 (rel: 0,0985) 0,0102790
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.1 fval: 0.006683977541420521 rhs: 0.004344037323277904 cdd: 0.12629396138966492
    [run-main-0] INFO breeze.optimize.StrongWolfeLineSearch - Line search t: 0.04629320022261909 fval: 0.004179143967394738 rhs: 0.004344058639229679 cdd: -0.0021041013125197186
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:23:50:516] Taking step 42. Step Size: 0,04629. Val and Grad Norm: 0,00417914 (rel: 0,0380) 0,565816
    [run-main-0] INFO neuroflow.nets.NFLBFGS - [06.05.2017 02:24:44:975] Taking step 43. Step Size: 1,000. Val and Grad Norm: 0,000482316 (rel: 0,885) 4,83443
    [run-main-0] INFO neuroflow.nets.NFLBFGS - Converged because Error function is sufficiently minimal.
    Vector(0.9996578942094551, 0.0012881726874004664, 0.001999699986111932)
    Vector(0.001558946909643395, 0.9996069943591613, 2.2674086144399572E-8)
    Vector(4.6652309999869356E-4, 1.1903004884534574E-4, 0.9999830386917378)
    Vector(0.9996578942094551, 0.0012881726874004664, 0.001999699986111932)
    Vector(0.001558946909643395, 0.9996069943591613, 2.2674086144399572E-8)
    Vector(4.6652309999869356E-4, 1.1903004884534574E-4, 0.9999830386917378)
    ...
    Vector(0.9996578942094551, 0.0012881726874004664, 0.001999699986111932)
    Vector(0.001558946909643395, 0.9996069943591613, 2.2674086144399572E-8)
    Vector(4.6652309999869356E-4, 1.1903004884534574E-4, 0.9999830386917378)

    Vector(0.39102338166639117, 0.01850119168167615, 0.03268279230327175) <- generalityPeek
    Vector(0.952033626753371, 0.01966032830147877, 9.146432325148978E-4)  <- generalityPeek
    [success] Total time: 7633 s, completed 06.05.2017 02:24:45


 */
