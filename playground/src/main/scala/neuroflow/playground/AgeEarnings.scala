package neuroflow.playground


import neuroflow.core.Activator.Sigmoid
import neuroflow.core.{Output, Hidden, Input, Network}

import scala.io.Source

/**
  * @author bogdanski
  * @since 03.01.16
  */


object AgeEarnings {
  def apply = {
    val src = Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("file/adult.txt")).getLines().map(_.split(",")).flatMap(k => {
      (if (k.size > 14) Some(k(14)) else None).map { over50k => (k(0).toDouble, if (over50k.equals(" >50K")) 1.0 else 0.0) }
    }).toList

    val train = src.take(1000)
    val test = src.drop(1000)
    val network = Network(Input(1) ::
      Hidden(20, Sigmoid.apply) ::
      Output(1, Sigmoid.apply) :: Nil)
    val maxAge = train.map(_._1).sorted.reverse.head
    val xs = train.map(a => Seq(a._1 / maxAge))
    val ys = train.map(a => Seq(a._2))
    network.train(xs, ys, 0.05, 0.001, 5000)

    val result = Range.Double(0.0, 1.1, 0.1).map(k => (k * maxAge, network.evaluate(Seq(k))))
    result.foreach { r =>
      println(s"Age ${r._1} earning >50K ${r._2.head}")
    }

    def testModel(age: Int): Unit = {
      val (a, b) = test.filter(_._1 == age).partition(k => k._2 == 1)
      if (a.size + b.size > 0)
        println(s"Test data, age $age, count >50K: ${a.size} (= ${a.size/((a.size + b.size)/100.0)}%), count <=50K: ${b.size}, overall: ${a.size + b.size}")
    }

    Range(10, 95, 5).foreach(testModel)
  }


  /*
      After 5000 iterations the model predicted:

      Age 0.0 earning >50K  0.000000513427763447596
      Age 9.0 earning >50K  0.0001.281087951788469
      Age 18.0 earning >50K 0.009092619353279468
      Age 27.0 earning >50K 0.11078107734768486
      Age 36.0 earning >50K 0.30148562762681846
      Age 45.0 earning >50K 0.3826525347737731
      Age 54.0 earning >50K 0.35164129029417657
      Age 63.0 earning >50K 0.2688573971829743
      Age 72.0 earning >50K 0.18102858098849597
      Age 81.0 earning >50K 0.11235410766655485
      Age 90.0 earning >50K 0.06688813747692092
   */
}
