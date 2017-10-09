package neuroflow.nets.gpu.cuda

/**
  * @author dlwh
  **/
case class Dim3(x: Int, y: Int = 1, z: Int = 1) {
  def asArray = Array(x, y, z)
}

object Dim3 {

  val default: Dim3 = Dim3(1)

  implicit def fromInt(x: Int): Dim3 = new Dim3(x)
  implicit def fromTuple2(x: (Int, Int)): Dim3 = new Dim3(x._1, x._2)
  implicit def fromTuple3(x: (Int, Int, Int)): Dim3 = new Dim3(x._1, x._2, x._3)

}
