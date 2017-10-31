package neuroflow.cuda

import neuroflow.common.Logs
import neuroflow.core.{Network, TypeSize}

/**
  * @author bogdanski
  * @since 31.10.17
  */
object GcThreshold extends Logs {

  private var threshold: Long = 100L * 1024L * 1024L // 100 MB init

  def apply(): Long = threshold

  def set(t: Long): Unit = {
    threshold = t
    debug(f"GcThreshold = $t Bytes (≈ ${ t / 1024.0 / 1024.0 }%.6g MB)")
  }

  def set[V](n: Network[V, _, _], f: Int)(implicit ts: TypeSize[V]): Unit = {
    val ws = n.weights.map(_.size).sum
    val as = n.layers.map(_.neurons).sum
    val t = ((3 * ws) + (as * f)) * ts()
    threshold = t
    debug(f"GcThreshold = $t Bytes (≈ ${ t / 1024.0 / 1024.0 }%.6g MB)")
  }

}
