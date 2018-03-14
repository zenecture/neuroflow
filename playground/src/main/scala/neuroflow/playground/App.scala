package neuroflow.playground

/**
  * @author bogdanski
  * @since 03.01.16
  */

object App extends App {

  println("Run example (1-19): ")

  scala.io.StdIn.readInt() match {

    case  1 => XOR.apply
    case  2 => ShallowNet.apply
    case  3 => TrendDetection.apply
    case  4 => AgeEarnings.apply
    case  5 => ImageRecognition.apply
    case  6 => DigitRecognition.apply
    case  7 => Sinusoidal.apply
    case  8 => AudioRecognition.apply
    case  9 => TextClassification.apply
    case 10 => TextClassification.test
    case 11 => Sequences.apply
    case 12 => MovieCluster.apply
    case 13 => MovieCluster.find
    case 14 => ParityCluster.apply
    case 15 => PokeMonCluster.apply
    case 16 => DistributedTraining.coordinator
    case 17 => DistributedTraining.executors
    case 18 => Word2Vec.apply
    case 19 => ConvViz.apply

    case  _ => sys.exit()

  }

}
