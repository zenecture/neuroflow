import neuroflow.application.plugin.IO.{File, Json}
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.{Hidden, Input, Network, Output}
import neuroflow.nets.DefaultNetwork.constructor
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * @author bogdanski
  * @since 09.01.16
  */
class IoTest extends Specification {
  def is: SpecStructure = s2"""

    This spec will test IO related functionality.

    It should:
      - Serialize a net                       $serialize
      - Deserialize a net                     $deserialize
      - Write to file                         $writeToFile
      - Read from file                        $readFromFile

  """

  val layers = Input(2) :: Hidden(3, Sigmoid.apply) :: Output(2, Sigmoid.apply) :: Nil
  val measure = {
    import neuroflow.core.WeightProvider.zeroWeights
    Network(layers)
  }
  val asJson = "{\n  \"$type\": \"scala.collection.Seq[breeze.linalg.DenseMatrix[scala.Double]]\",\n  \"elems\": [\n    {\n    \"$type\": \"breeze.linalg.DenseMatrix$mcD$sp\",\n    \"rows\": 2,\n    \"cols\": 3,\n    \"data\": [\n      0.0,\n      0.0,\n      0.0,\n      0.0,\n      0.0,\n      0.0\n    ],\n    \"offset\": 0,\n    \"majorStride\": 2,\n    \"isTranspose\": false\n  },\n    {\n    \"$type\": \"breeze.linalg.DenseMatrix$mcD$sp\",\n    \"rows\": 3,\n    \"cols\": 2,\n    \"data\": [\n      0.0,\n      0.0,\n      0.0,\n      0.0,\n      0.0,\n      0.0\n    ],\n    \"offset\": 0,\n    \"majorStride\": 3,\n    \"isTranspose\": false\n  }\n  ]\n}"
  val file = "/Users/felix/github/unversioned/savednets/net.json"

  def serialize = {
    val serialized = Json.write(measure)
    serialized === asJson
  }

  def deserialize = {
    implicit val wp = Json.read(asJson)
    val deserialized = Network(layers)
    deserialized.weights === measure.weights
  }

  def writeToFile = {
    File.write(measure, file)
    success
  }

  def readFromFile = {
    implicit val wp = File.read(file)
    val net = Network(layers)
    net.weights === measure.weights
  }

}