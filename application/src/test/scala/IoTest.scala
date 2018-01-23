import neuroflow.application.plugin.IO._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import shapeless._

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

  val layers = Input(2) :: Dense(3, Sigmoid) :: Output(2, Sigmoid) :: HNil

  val measure: FFN[Double] = {
    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].static(0.0)
    Network(layers)
  }

  def io = {
    val serialized = Json.write(measure.weights)
    implicit val wp = Json.read[Double](serialized)
    val deserialized = Network(layers)

    deserialized.weights.toArray.map(_.toArray) === measure.weights.toArray.map(_.toArray)
  }

}
