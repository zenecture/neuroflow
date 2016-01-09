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
    val selector: Int => Boolean = _ < 255
    (0 to 9) map (i => extractBinary(getFile(path + s"$i.png"), selector).grouped(100).toList)
  }

  def apply = {
    val sets = ('a' to 'h') map (c => getDigitSet(s"img/digits/$c/"))
    val nets = sets.head.head.indices.par.map { segment =>
      val fn = Sigmoid.apply
      val xs = sets dropRight 1 flatMap { s => (0 to 9) map { digit => s(digit)(segment) } }
      val ys = sets dropRight 1 flatMap { m => (0 to 9) map { digit => Seq(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).updated(digit, 1.0) } }
      val net = Network(Input(xs.head.size) :: Hidden(50, fn) :: Output(10, fn) :: Nil, Settings(numericGradient = true, Δ = 0.00001, verbose = true))
      net.train(xs, ys, 2.0, 0.001, 200)
      net
    }

    val setsResult = sets map { set => set map { d => d flatMap { xs => nets map { _.evaluate(xs) } } reduce((a, b) => a.zip(b) map (l => l._1 + l._2)) map (end => end / nets.size) } }

    ('a' to 'h') zip setsResult foreach { pair =>
      val (char, res) = pair
      println(s"set $char:")
      (0 to 9) foreach { digit => println(s"$digit classified as " + res(digit).indexOf(res(digit).max)) }
    }

  }

}


/*

[INFO] [09.01.2016 13:00:18:190] [ForkJoinPool-1-worker-13] Took 200 iterations of 200 with error 0.2514897816501956  0.0012656955183186303  ... (10 total)
[INFO] [09.01.2016 13:00:20:629] [ForkJoinPool-1-worker-11] Took 200 iterations of 200 with error 9.645298738455374E-4  0.0010791357253543453  ... (10 total)
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
[success] Total time: 1397 s, completed 09.01.2016 13:00:20


 */