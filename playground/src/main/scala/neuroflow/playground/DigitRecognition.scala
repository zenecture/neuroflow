package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.application.processor.Util._
import neuroflow.common.VectorTranslation._
import neuroflow.common.~>
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 04.01.16
  */
object DigitRecognition {

  /*

   Here the goal is to classify digits from unknown font family 'h'.
   Feel free to read this article for the full story:
      http://znctr.com/blog/digit-recognition

  */

  def getDigitSet(path: String) = {
    val selector: Int => Boolean = _ < 255
    (0 to 9) map (i => extractBinary(getResourceFile(path + s"$i.png"), selector).grouped(100).toVector)
  }

  def apply = {

    val sets = ('a' to 'h') map (c => getDigitSet(s"img/digits/$c/").toVector)
    val nets = sets.head.head.indices.par.map { segment =>
      val fn = Sigmoid
      val settings = Settings(
        learningRate = { case _ => 0.05 },
        precision = 0.1, iterations = 5000,
        regularization = Some(KeepBest))
      val xs = sets.dropRight(1).flatMap { s => (0 to 9) map { digit => s(digit)(segment) } }
      val ys = sets dropRight 1 flatMap { m => (0 to 9) map { digit => ->(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).toScalaVector.updated(digit, 1.0) } }
      val net = Network(Input(xs.head.size) :: Hidden(50, fn) :: Output(10, fn) :: HNil, settings)
      net.train(xs.map(_.dv), ys.map(_.dv))
      net
    }

    val setsResult = sets map { set => set map { d => d flatMap { xs => nets map { _.evaluate(xs.dv) } } reduce(_ + _) map (end => end / nets.size) } }

    ('a' to 'h') zip setsResult foreach {
      case (char, res) =>
        ~> (println(s"set $char:")) next (0 to 9) foreach { digit => println(s"$digit classified as " + res(digit).toScalaVector.indexOf(res(digit).max)) }
    }

  }

}


/*

    [scala-execution-context-global-67] INFO neuroflow.nets.DefaultNetwork - [27.07.2017 23:57:07:764] Taking step 1808 - Mean Error 0,100076 - Error Vector 0.31790819492664857  0.028343763942545036  0.042212692573426476  ... (10 total)
    [scala-execution-context-global-67] INFO neuroflow.nets.DefaultNetwork - [27.07.2017 23:57:07:768] Took 1809 iterations of 5000 with Mean Error = 0,0999966
    [scala-execution-context-global-67] INFO neuroflow.nets.DefaultNetwork - [27.07.2017 23:57:07:768] Applying KeepBest strategy. Best test error so far: 0,100076.
    set a:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set b:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set c:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set d:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set e:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set f:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set g:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    set h:
    0 classified as 0
    1 classified as 1
    2 classified as 2
    3 classified as 3
    4 classified as 4
    5 classified as 5
    6 classified as 6
    7 classified as 7
    8 classified as 8
    9 classified as 9
    [success] Total time: 23 s, completed 27.07.2017 23:57:07


 */
