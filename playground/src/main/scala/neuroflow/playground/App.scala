package neuroflow.playground

/**
  * @author bogdanski
  * @since 03.01.16
  */

object App extends App {

  print("Run example (1-13): ")

  13 match {
    case  1 => XOR.apply
    case  2 => SigGap.apply
    case  3 => Trending.apply
    case  4 => AgeEarnings.apply
    case  5 => ImageRecognition.apply
    case  6 => DigitRecognition.apply
    case  7 => Sinusoidal.apply
    case  8 => AudioFileClassification.apply
    case  9 => LanguageProcessing.apply
               LanguageProcessing.test
    case 10 => Sequences.apply
    case 11 => MovieSimilarity.apply
               MovieSimilarity.find
    case 12 => ParityCluster.apply
    case 13 => PokeMonCluster.apply

    case  _ => sys.exit()
  }

}
