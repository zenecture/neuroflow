package neuroflow.nets.gpu.cuda

import breeze.linalg.DenseMatrix
import breeze.numerics.sin
import jcuda.jcublas._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * @author bogdanski
  * @since 21.09.17
  */
class CuMatrixTest  extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test matrix operations on GPU.

      - Mult two matrices                                $mult
      - AddSub two matrices                              $addSub
      - Convert DenseMatrix <-> CuMatrix                 $convert
      - Bench with CPU                                   $bench

  """

  def mult = {

    implicit val handle = new cublasHandle
    JCublas2.cublasCreate(handle)

    val a = CuMatrix.ones[Double](4, 4)
    val b = CuMatrix.ones[Double](4, 4)
    val c = a * b.t
    val d = (a += 4.0) *:* (b += 4.0)
    val e = a.reshape(1, 16) *:* b.reshape(1, 16)
    println(a)
    println(b)
    println(c)
    println(d)
    println(e)
    a.release()
    b.release()
    c.release()
    d.release()
    e.release()

    success

  }

  def addSub = {

    implicit val handle = new cublasHandle
    JCublas2.cublasCreate(handle)

    val a = CuMatrix.ones[Double](4, 4)
    val b = CuMatrix.ones[Double](4, 4)
    val c = a + b
    val d = b - a
    val e = d += 7.0
    val f = sin(e)
    println(a)
    println(b)
    println(c)
    println(d)
    println(e)
    a.release()
    b.release()
    c.release()
    d.release()
    e.release()
    f.release()

    success

  }

  def convert = {

    implicit val handle = new cublasHandle
    JCublas2.cublasCreate(handle)

    val a = CuMatrix.ones[Double](4, 4)
    val b = a.toDense
    val c = CuMatrix.fromDense(b)

    println(a)
    println(b)
    println(c)

    success

  }

  def bench = {

    implicit val handle = new cublasHandle
    JCublas2.cublasCreate(handle)

    Seq(50, 500, 5000).foreach(n => benchWithCreation(n))
    Seq(50, 500, 5000).foreach(n => benchWithoutCreation(n))

    def benchWithCreation(n: Int) = {

      def withTimer[B](f: () => B): (Long, B) = {
        val n1 = System.nanoTime()
        val r = f()
        val n2 = System.nanoTime()
        n2 - n1 -> r
      }

      val c = withTimer(() => {
        val a = CuMatrix.rand(n, n)
        val b = CuMatrix.rand(n, n)
        (a, b, a * b)
      })
      c._2._1.release()
      c._2._2.release()
      c._2._3.release()

      val C = withTimer(() => {
        val A = DenseMatrix.rand[Double](n, n)
        val B = DenseMatrix.rand[Double](n, n)
        A * B
      })

      println(s"++ c: ${c._1}")
      println(s"++ C: ${C._1}")
      println(s"++ n: $n")

    }

    def benchWithoutCreation(n: Int) = {

      def withTimer[B](f: () => B): (Long, B) = {
        val n1 = System.nanoTime()
        val r = f()
        val n2 = System.nanoTime()
        n2 - n1 -> r
      }

      val a = CuMatrix.rand(n, n)
      val b = CuMatrix.rand(n, n)
      val c = withTimer(() => a * b)
      a.release()
      b.release()
      c._2.release()

      val A = DenseMatrix.rand[Double](n, n)
      val B = DenseMatrix.rand[Double](n, n)
      val C = withTimer(() => A * B)

      println(s"-- c: ${c._1}")
      println(s"-- C: ${C._1}")
      println(s"-- n: $n")

    }

    success

  }

}
