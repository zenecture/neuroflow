package neuroflow.playground

/**
  * @author bogdanski
  * @since 03.01.16
  */

object App extends App {

  println("Run example (1-17): ")

  scala.io.StdIn.readInt() match {
    case  1 => XOR.apply
    case  2 => SigGap.apply
    case  3 => Trending.apply
    case  4 => AgeEarnings.apply
    case  5 => ImageRecognition.apply
    case  6 => DigitRecognition.apply
    case  7 => Sinusoidal.apply
    case  8 => AudioRecognition.apply
    case  9 => LanguageProcessing.apply
    case 10 => LanguageProcessing.test
    case 11 => Sequences.apply
    case 12 => MovieCluster.apply
    case 13 => MovieCluster.find
    case 14 => ParityCluster.apply
    case 15 => PokeMonCluster.apply
    case 16 => DistributedTraining.coordinator
    case 17 => DistributedTraining.executor

    case  _ => sys.exit()
  }

}
