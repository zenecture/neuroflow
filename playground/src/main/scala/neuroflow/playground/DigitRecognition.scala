package neuroflow.playground

import neuroflow.application.classification.Image._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core._

/**
  * @author bogdanski
  * @since 04.01.16
  */
object DigitRecognition {

  def getDigitSet(path: String) = {
    val selector: Int => Boolean = c => c < 255
    (0 to 9) map (i => extractBinary(getFile(path + s"$i.png"), selector).grouped(100).toList)
  }

  def apply = {
    val setA = getDigitSet("img/digits/e/") // training
    val setB = getDigitSet("img/digits/b/") // training
    val setC = getDigitSet("img/digits/c/") // training
    val setD = getDigitSet("img/digits/d/") // training
    val setE = getDigitSet("img/digits/a/") // test
    val nets = (0 to (setA.head.size - 1)).par.map { segment =>
      val xs =
        ((0 to 9) map { digit => setA(digit)(segment) }) ++
        ((0 to 9) map { digit => setB(digit)(segment) }) ++
        ((0 to 9) map { digit => setC(digit)(segment) }) ++
        ((0 to 9) map { digit => setD(digit)(segment) })
      val ys = (0 to 9) map { digit => Seq(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).updated(digit, 1.0) }
      val fn = Sigmoid.apply
      val net = Network(Input(xs.head.size) :: Hidden(50, fn) :: Output(10, fn) :: Nil)
      net.train(xs, ys ++ ys ++ ys ++ ys, 2.0, 0.001, 30)
      net
    }

    val setAResult = setA map { d => d flatMap { xs => nets map { _.evaluate(xs) } } reduce((a, b) => a.zip(b) map (l => l._1 + l._2)) map (end => end / nets.size) }
    val setBResult = setB map { d => d flatMap { xs => nets map { _.evaluate(xs) } } reduce((a, b) => a.zip(b) map (l => l._1 + l._2)) map (end => end / nets.size) }
    val setCResult = setC map { d => d flatMap { xs => nets map { _.evaluate(xs) } } reduce((a, b) => a.zip(b) map (l => l._1 + l._2)) map (end => end / nets.size) }
    val setDResult = setD map { d => d flatMap { xs => nets map { _.evaluate(xs) } } reduce((a, b) => a.zip(b) map (l => l._1 + l._2)) map (end => end / nets.size) }
    val setEResult = setE map { d => d flatMap { xs => nets map { _.evaluate(xs) } } reduce((a, b) => a.zip(b) map (l => l._1 + l._2)) map (end => end / nets.size) }

    println("set A:")
    (0 to 9) foreach { digit => println(s"$digit classified as " + setAResult(digit).indexOf(setAResult(digit).max)) }
    println("set B:")
    (0 to 9) foreach { digit => println(s"$digit classified as " + setBResult(digit).indexOf(setBResult(digit).max)) }
    println("set C:")
    (0 to 9) foreach { digit => println(s"$digit classified as " + setCResult(digit).indexOf(setCResult(digit).max)) }
    println("set D:")
    (0 to 9) foreach { digit => println(s"$digit classified as " + setDResult(digit).indexOf(setDResult(digit).max)) }
    println("set E:")
    (0 to 9) foreach { digit => println(s"$digit classified as " + setEResult(digit).indexOf(setEResult(digit).max)) }


  }

}


/*

set A: 0 classified as 0
set A: 1 classified as 1
set A: 2 classified as 2
set A: 3 classified as 3
set A: 4 classified as 4
set A: 5 classified as 5
set A: 6 classified as 6
set A: 7 classified as 7
set A: 8 classified as 8
set A: 9 classified as 9
---
set B: 0 classified as 0
set B: 1 classified as 1
set B: 2 classified as 2
set B: 3 classified as 3
set B: 4 classified as 4
set B: 5 classified as 5
set B: 6 classified as 6
set B: 7 classified as 7
set B: 8 classified as 8
set B: 9 classified as 0
---
set C: 0 classified as 0
set C: 1 classified as 1
set C: 2 classified as 2
set C: 3 classified as 3
set C: 4 classified as 4
set C: 5 classified as 5
set C: 6 classified as 6
set C: 7 classified as 7
set C: 8 classified as 8
set C: 9 classified as 9
---
set D: 0 classified as 0
set D: 1 classified as 1
set D: 2 classified as 2
set D: 3 classified as 0  <--- Error
set D: 4 classified as 4
set D: 5 classified as 5
set D: 6 classified as 6
set D: 7 classified as 7
set D: 8 classified as 8  <--- Error
set D: 9 classified as 4
[success] Total time: 618 s, completed 07.01.2016 12:42:15


 */