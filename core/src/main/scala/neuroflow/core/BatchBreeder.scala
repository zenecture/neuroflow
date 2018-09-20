package neuroflow.core

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import neuroflow.common.Logs

import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 24.03.18
  */

object BatchBreeder extends Logs {

  /**
    * Groups `xs` and targets `ys` into batches in parallel.
    * Memory = O(2n)
    */
  def breedFFN[V: ClassTag : Zero](xs: Seq[DenseVector[V]], ys: Seq[DenseVector[V]], batchSize: Int): Seq[(DenseMatrix[V], DenseMatrix[V])] = {

    val xsys = xs.zip(ys).grouped(batchSize).zipWithIndex.toSeq.par.map { case (xy, batchNo) =>

      val x = DenseMatrix.zeros[V](xy.size, xy.head._1.length)
      val y = DenseMatrix.zeros[V](xy.size, xy.head._2.length)

      (0 until x.rows).foreach { row =>
        (0 until x.cols).foreach { col =>
          x.update(row, col, xy(row)._1(col))
        }
      }

      (0 until y.rows).foreach { row =>
        (0 until y.cols).foreach { col =>
          y.update(row, col, xy(row)._2(col))
        }
      }

      debug(s"Bred Batch $batchNo.")

      x -> y

    }.seq

    xsys

  }

  /**
    * Groups `xs` and targets `ys` into batches in parallel.
    *  (Additionally, it returns a map from batch to batchSize,
    *   to take care for unevenly distributed batches.)
    * Memory = O(2n)
    */
  def breedCNN[V: ClassTag : Zero](xs: Seq[Tensor3D[V]], ys: Seq[DenseVector[V]], batchSize: Int): (Seq[(DenseMatrix[V], DenseMatrix[V])], Map[Int, Int]) = {

    val xsys = xs.zip(ys).grouped(batchSize).zipWithIndex.toSeq.par.map { case (xy, batchNo) =>

      val x = DenseMatrix.zeros[V](xy.head._1.matrix.rows, xy.head._1.matrix.cols * xy.size)
      val y = DenseMatrix.zeros[V](xy.size, xy.head._2.length)

      (0 until x.rows).foreach { row =>
        (0 until x.cols).foreach { col =>
          val b = col / xy.head._1.matrix.cols
          val c = col % xy.head._1.matrix.cols
          x.update(row, col, xy(b)._1.matrix(row, c))
        }
      }

      (0 until y.rows).foreach { row =>
        (0 until y.cols).foreach { col =>
          y.update(row, col, xy(row)._2(col))
        }
      }

      debug(s"Bred Batch $batchNo.")

      (x -> y) -> xy.size

    }.seq

    xsys.map(_._1) -> xsys.zipWithIndex.map(b => b._2 -> b._1._2).toMap

  }

}

