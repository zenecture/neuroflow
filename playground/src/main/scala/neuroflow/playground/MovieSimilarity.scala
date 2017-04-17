package neuroflow.playground

import java.io.{File, FileOutputStream, PrintWriter}

import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Util._
import neuroflow.application.processor.{Extensions, Normalizer, Util}
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.LBFGSCluster._
import shapeless._

import scala.io.{Source, StdIn}

/**
  * @author bogdanski
  * @since 15.04.17
  */

object MovieSimilarity {

  case class Movie(id: Int, title: String, vec: Network.Vector)
  case class Rating(user: Int, movieId: Int, rating: Int)

  val netFile = "/Users/felix/github/unversioned/movies.nf"
  val clusterOutput = "/Users/felix/github/unversioned/clusters.txt"

  val dimensionLimit = 300
  val observationLimit = 50000

  val movies: List[Movie] =
    ~>(Source.fromFile(getResourceFile("file/ml-100k/u.item")).getLines.toList.take(dimensionLimit)).map { ms =>
      ms.map { line =>
        val r = line.replace("|", ";").split(";")
        Movie(r(0).toInt, r(1), ζ(ms.size).updated(r(0).toInt - 1, 1.0))
      }
    }

  val observations: List[Rating] = Source.fromFile(getResourceFile("file/ml-100k/u.data"))
    .getLines.map(_.split("\t")).map(r => Rating(r(0).toInt, r(1).toInt, r(2).toInt)).toList

  val layout = Input(movies.size) :: Cluster(3, Linear) :: Output(movies.size, Sigmoid) :: HNil

  def apply = {

    import neuroflow.core.FFN.WeightProvider._
    import Extensions.SeqLikeVector

    val topByUser = observations.take(observationLimit).filter(_.rating == 5).groupBy(_.user).map {
      case (user, ratings) =>
        val vecs = ratings.flatMap(r => if (r.movieId <= dimensionLimit) Some(movies(r.movieId - 1).vec) else None)
        Util.shuffle(vecs).map {
          case (k, v) => k -> Normalizer.MaxUnit(v.reduce(_ + _))
        }
    }.toList.flatten

    println("Training samples: " + topByUser.size)

    val net = Network(layout, Settings(iterations = 25))

    net.train(topByUser.map(_._1), topByUser.map(_._2))

    IO.File.write(net, netFile)

  }

  def find = {

    val net = {
      implicit val wp = IO.File.read(netFile)
      Network(layout, Settings())
    }

    val res = movies.map(m => m.copy(vec = Normalizer.UnitVector(net.evaluate(m.vec))))

    val outputFile = ~>(new File(clusterOutput)).io(_.delete)
    ~>(new PrintWriter(new FileOutputStream(outputFile, true))).io { writer =>
      res.foreach(v => writer.println(prettyPrint(v.vec, ";") + ";" + v.title))
    }.io(_.close)

    var findId: Int = 0
    while ({ print("Find movieId: "); findId = StdIn.readInt(); findId >= 0 }) {
      val target = res(findId)
      val all = res.map {
        case Movie(_, title, vec) =>
          (title, Extensions.cosineSimilarity(target.vec, vec))
      }.sortBy(_._2)
      val best = all.reverse.take(10)
      val worst = all.take(10)
      println("The 10 most (un-)similar movies for: " + target.title)
      best.foreach(m => println(m))
      println("...")
      worst.foreach(m => println(m))
      println()
      println()
    }

  }

}


/*

    See:
        - resources/file/ml-100k/MovieCloud.png
        - resources/file/ml-100k/MovieCloudL.png

    Find movieId: 36
    The 10 most (un-)similar movies for: Nadja (1994)
    (Nadja (1994),1.0000000000000002)
    (Kansas City (1996),0.54186693884391)
    (Free Willy 2: The Adventure Home (1995),0.37849939209395833)
    (Mimic (1997),0.3781999525612457)
    (Mad Love (1995),0.24935136110925749)
    (Twelve Monkeys (1995),-0.01001494196088354)
    (Batman & Robin (1997),-0.0950935761314235)
    (Rock, The (1996),-0.09623026204644874)
    (Promesse, La (1996),-0.10399911124969907)
    (Amadeus (1984),-0.12995581089852)
    ...
    (Pink Floyd - The Wall (1982),-0.9990339133043322)
    (Nikita (La Femme Nikita) (1990),-0.9983140999393241)
    (Jude (1996),-0.9979989237940924)
    (Horseman on the Roof, The (Hussard sur le toit, Le) (1995),-0.9968073903210092)
    (Cinema Paradiso (1988),-0.9943456650005529)
    (Rumble in the Bronx (1995),-0.9941455598932651)
    (Alien (1979),-0.9919619374747143)
    (Madness of King George, The (1994),-0.9910138507914289)
    (Die Hard 2 (1990),-0.9878334851355521)
    (Last of the Mohicans, The (1992),-0.9871419174832422)


    Find movieId: 248
    The 10 most (un-)similar movies for: Austin Powers: International Man of Mystery (1997)
    (Austin Powers: International Man of Mystery (1997),1.0)
    (Stargate (1994),0.9999447952506219)
    (Supercop (1992),0.9995768787253511)
    (Batman Returns (1992),0.9991786113588649)
    (Natural Born Killers (1994),0.9991646064003881)
    (Ace Ventura: Pet Detective (1994),0.9987625132526217)
    (Mars Attacks! (1996),0.9985953803999155)
    (Ref, The (1994),0.9985449290579209)
    (Aristocats, The (1970),0.9984795259166293)
    (Maverick (1994),0.9979267681441143)
    ...
    (Nadja (1994),-0.9416555272158736)
    (Kansas City (1996),-0.44930542326722545)
    (Mimic (1997),-0.377332942648324)
    (Promesse, La (1996),-0.15454584674877161)
    (Free Willy 2: The Adventure Home (1995),-0.10162428851114265)
    (Mad Love (1995),-0.07003133770064952)
    (Faster Pussycat! Kill! Kill! (1965),0.03533770752223712)
    (Theodore Rex (1995),0.32866995133890714)
    (Twelve Monkeys (1995),0.34579731041018585)
    (Wizard of Oz, The (1939),0.364071986200786)

 */
