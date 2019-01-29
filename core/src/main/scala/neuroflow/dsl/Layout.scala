package neuroflow.dsl

import neuroflow.core._
import neuroflow.dsl

/**
  * @author bogdanski
  * @since 27.01.18
  */


/**
  * The layout describes the neural net as a linear graph
  * structure. It is implemented as a heterogenous list,
  * allowing compile-time checks for valid compositions.
  */
trait Layout extends Serializable {

  /** Prepends this layout with a new layer `head`. */
  def ::[H <: Layer[_]](head: H): H :: this.type = dsl.::(head, tail = this)

}


case class ::[+H <: Layer[_], +T <: Layout](head: H, tail: T) extends Layout



object Layout {

  implicit class LayoutTraversable(l: Layout) {

    def foreach[V](f: PartialFunction[Layer[V], Unit]): Unit = trav(l)(f)(noOp)

    def map[A, V](f: PartialFunction[Layer[V], A]): Seq[A] = {
      val bldr = Seq.newBuilder[A]
      trav[V](l) { case l: Layer[V] => bldr += f(l) } (noOp)
      bldr.result()
    }

    def toSeq[V]: Seq[Layer[V]] = {
      val bldr = Seq.newBuilder[Layer[V]]
      trav[V](l) { case l: Layer[V] => bldr += l } (noOp)
      bldr.result()
    }

    def toLossFunction[V]: LossFunction[V] = {
      val bldr = Seq.newBuilder[LossFunction[V]]
      trav[V](l){ case _ => } (lf => bldr += lf)
      bldr.result().head
    }

    private def trav[V](xs: Layout)(f: PartialFunction[Layer[V], Unit])(g: LossFunction[V] => Unit): Unit = xs match {
      case (head: Layer[V]) :: tail =>
        f(head)
        trav(tail)(f)(g)
      case l: LossFunction[V] => g(l)
    }

    private val noOp: LossFunction[_] => Unit = { _ => }

  }

}

