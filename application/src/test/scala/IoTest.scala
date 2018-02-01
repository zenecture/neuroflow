import neuroflow.application.plugin.IO._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core._
import neuroflow.dsl._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * @author bogdanski
  * @since 09.01.16
  */
class IoTest extends Specification {

  sequential // IO race conditions will occur otherwise

  def is: SpecStructure = s2"""

    This spec will test IO related functionality.

    It should:
      - Read and write weights                $io

  """

  import neuroflow.nets.cpu.DenseNetwork._

  val layers = Vector(2) :: Dense(3, Sigmoid) :: Dense(2, Sigmoid) :: SquaredMeanError()

  val measure: FFN[Double] = {
    implicit val wp = neuroflow.core.WeightProvider[Double].static(0.0)
    Network(layers)
  }

  def io = {
    val serialized = Json.writeWeights(measure.weights)
    implicit val wp = Json.readWeights[Double](serialized)
    val deserialized = Network(layers)

    deserialized.weights.toArray.map(_.toArray) === measure.weights.toArray.map(_.toArray)
  }

}
