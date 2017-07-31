package neuroflow.playground

import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Extensions.VectorOps
import neuroflow.application.processor.Util._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core._
import shapeless._

import scala.io.Source

/**
  * @author bogdanski
  * @since 15.04.17
  */

object MovieRecommender {

  // Needs a lot of RAM. 16G or more would be good.

  case class Movie(id: Int, title: String)
  case class Rating(user: Int, movieId: Int, rating: Int)

  val file = "/Users/felix/github/unversioned/movierec.nf"

  val movies: List[Movie] =
    ~>(Source.fromFile(getResourceFile("file/ml-100k/u.item")).getLines.toList).map { ms =>
      ms.map { line =>
        val r = line.replace("|", ";").split(";")
        Movie(r(0).toInt, r(1))
      }
    }

  val ratings: List[Rating] = Source.fromFile(getResourceFile("file/ml-100k/u.data"))
    .getLines.map(_.split("\t")).map(r => Rating(r(0).toInt, r(1).toInt, r(2).toInt)).toList

  val tts = ratings.count(_.rating == 5)
  val fls = ratings.count(_.rating == 1)

  val lebowski = movies.find(_.title.contains("Lebowski"))
  val allLeb = ratings.filter(_.movieId == lebowski.get.id)
  val avgLeb = allLeb.map(_.rating).sum.toDouble / allLeb.size.toDouble

  val toy = movies.find(_.title.contains("Toy Story"))
  val allToy = ratings.filter(_.movieId == toy.get.id)
  val avgToy = allToy.map(_.rating).sum.toDouble / allToy.size.toDouble

  val all = movies.map(m => ratings.count(_.movieId == m.id))
  val avg = all.sum.toDouble / movies.size.toDouble

  println("allLeb.size =  " + allLeb.size)
  println("avgLeb = " + avgLeb)
  println("allToy.size =  " + allToy.size)
  println("avgToy = " + avgToy)
  println("avg = " + avg)

  println(s"tts: $tts, fls: $fls")

  val xs = ratings.groupBy(_.user).map {
    case (userId, rs) => rs.map { r =>
      ζ(movies.size).updated(r.movieId - 1, r.rating.toDouble * 0.2)
    }.reduce(_ + _)
  }.toList

  val layout = Input(movies.size) ::
    Hidden(50, Sigmoid) ::
    Hidden(25, Sigmoid) ::
    Hidden(12, Sigmoid) ::
    Hidden(25, Sigmoid) ::
    Hidden(50, Sigmoid) ::
    Output(movies.size, Sigmoid) :: HNil

  import neuroflow.nets.DefaultNetwork._

  def apply = {

    import neuroflow.core.FFN.WeightProvider._

    val net = Network(layout,
      Settings(iterations = 500,
        learningRate = {
          case iter if iter <= 10              =>  0.5
          case iter if iter  > 10 && iter < 50 =>  0.5
          case iter                            =>  0.5
        }, precision = 1E-3))

    net.train(xs, xs)

    IO.File.write(net, file)

  }

  def eval = {

    implicit val wp = IO.File.read(file)
    val net = Network(layout)

    xs.foreach { x =>
      val z = x.zipWithIndex.flatMap {
        case (i, k) if i == 0 => Some(k)
        case _ => None
      }

      val o = x.zipWithIndex.flatMap {
        case (i, k) if i > 0.6 => Some(movies.find(_.id == k + 1).get.title -> i)
        case _ => None
      }

      val t = net.evaluate(x).map(i => i / 0.2)
      val all = z.map { i => i -> t(i) }.sortBy(_._2).reverse.map {
        case (i, rat) => movies.find(_.id == i + 1).get.title -> rat
      }

      val (top, flop) = (all.take(5), all.takeRight(5))

      println(s"rated: $o")
      println(s"top: $top")
      println(s"flop: $flop")
      println()
    }

  }

}

/*

      [run-main-1] INFO neuroflow.nets.DefaultNetwork - [28.07.2017 11:18:41:246] Taking step 498 - Mean Error 11,7208 - Error Vector 65.86727054243158  28.528291046558593  16.555067671134474  ... (1682 total)
      [run-main-1] INFO neuroflow.nets.DefaultNetwork - [28.07.2017 11:18:42:029] Taking step 499 - Mean Error 11,6574 - Error Vector 71.06953020115836  28.547004817203906  17.017967952383913  ... (1682 total)
      [run-main-1] INFO neuroflow.nets.DefaultNetwork - [28.07.2017 11:18:42:854] Took 500 iterations of 500 with Mean Error = 11,6613




                   _   __                      ________
                  / | / /__  __  ___________  / ____/ /___ _      __
                 /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
               /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


               Version 0.701

               Identifier: joz
               Network: neuroflow.nets.DefaultNetwork
               Layout: [1682 In, 50 H (σ), 25 H (σ), 12 H (σ), 25 H (σ), 50 H (σ), 1682 Out (σ)]
               Number of Weights: 171300




      rated: Vector((Get Shorty (1995),0.8), (Seven (Se7en) (1995),0.8), (Braveheart (1995),0.8), (Taxi Driver (1976),1.0), (Apollo 13 (1995),0.8), (Belle de jour (1967),0.8), (Crumb (1994),1.0), (Strange Days (1995),0.6000000000000001), (Exotica (1994),1.0), (Ed Wood (1994),0.8), (Hoop Dreams (1994),0.8), (Star Wars (1977),0.8), (Professional, The (1994),0.6000000000000001), (Pulp Fiction (1994),0.6000000000000001), (Three Colors: Red (1994),1.0), (Three Colors: Blue (1993),1.0), (Three Colors: White (1994),1.0), (Shawshank Redemption, The (1994),0.6000000000000001), (What's Eating Gilbert Grape (1993),0.8), (Forrest Gump (1994),0.8), (Four Weddings and a Funeral (1994),0.8), (Mask, The (1994),0.6000000000000001), (Maverick (1994),0.6000000000000001), (Hudsucker Proxy, The (1994),0.8), (Searching for Bobby Fischer (1993),0.8), (Blade Runner (1982),0.8), (Nightmare Before Christmas, The (1993),0.6000000000000001), (True Romance (1993),0.6000000000000001), (Terminator 2: Judgment Day (1991),0.6000000000000001), (Silence of the Lambs, The (1991),0.8), (Citizen Kane (1941),1.0), (2001: A Space Odyssey (1968),1.0), (Monty Python and the Holy Grail (1974),0.8), (Empire Strikes Back, The (1980),0.8), (Princess Bride, The (1987),0.8), (Raiders of the Lost Ark (1981),0.8), (Brazil (1985),1.0), (Good, The Bad and The Ugly, The (1966),0.8), (Clockwork Orange, A (1971),1.0), (Apocalypse Now (1979),0.8), (Return of the Jedi (1983),0.8), (GoodFellas (1990),1.0), (Alien (1979),0.8), (Army of Darkness (1993),0.6000000000000001), (Psycho (1960),1.0), (Blues Brothers, The (1980),0.8), (Full Metal Jacket (1987),0.8), (Amadeus (1984),1.0), (Sting, The (1973),0.8), (Terminator, The (1984),0.8), (Graduate, The (1967),1.0), (Nikita (La Femme Nikita) (1990),0.6000000000000001), (Shining, The (1980),1.0), (Groundhog Day (1993),0.6000000000000001), (Unforgiven (1992),0.8), (Young Frankenstein (1974),1.0), (This Is Spinal Tap (1984),1.0), (M*A*S*H (1970),0.8), (Unbearable Lightness of Being, The (1988),0.8), (Pink Floyd - The Wall (1982),0.8), (When Harry Met Sally... (1989),0.8), (Star Trek: The Wrath of Khan (1982),0.6000000000000001), (Sneakers (1992),0.6000000000000001), (Contact (1997),0.6000000000000001), (Chasing Amy (1997),0.8), (English Patient, The (1996),0.8), (Scream (1996),0.6000000000000001), (Schindler's List (1993),1.0), (Everyone Says I Love You (1996),0.6000000000000001), (Boogie Nights (1997),0.8), (One Flew Over the Cuckoo's Nest (1975),1.0), (Clueless (1995),0.6000000000000001), (Batman (1989),0.6000000000000001), (To Kill a Mockingbird (1962),1.0), (Harold and Maude (1971),0.8), (Duck Soup (1933),1.0), (Heathers (1989),0.8), (Forbidden Planet (1956),0.8), (Butch Cassidy and the Sundance Kid (1969),0.8), (Carrie (1976),0.6000000000000001), (Short Cuts (1993),1.0), (Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963),1.0), (Some Like It Hot (1959),0.8), (Casablanca (1942),1.0), (Sunset Blvd. (1950),0.8), (It's a Wonderful Life (1946),0.6000000000000001), (Rebel Without a Cause (1955),1.0), (Wings of Desire (1987),1.0), (Third Man, The (1949),1.0), (Annie Hall (1977),1.0), (Miller's Crossing (1990),1.0), (Deer Hunter, The (1978),0.8), (Cool Hand Luke (1967),1.0), (Heavenly Creatures (1994),0.8), (Night of the Living Dead (1968),0.6000000000000001), (Cook the Thief His Wife & Her Lover, The (1989),0.8), (Paths of Glory (1957),1.0), (Seventh Seal, The (Sjunde inseglet, Det) (1957),1.0), (Touch of Evil (1958),1.0), (Chinatown (1974),1.0), (M (1931),0.8), (Pump Up the Volume (1990),0.8), (Fried Green Tomatoes (1991),0.6000000000000001), (Paris, Texas (1984),0.8), (Cape Fear (1962),0.6000000000000001), (Cat People (1982),0.6000000000000001), (Nosferatu (Nosferatu, eine Symphonie des Grauens) (1922),0.8), (Sex, Lies, and Videotape (1989),0.6000000000000001), (Strictly Ballroom (1992),0.6000000000000001), (Real Genius (1985),0.8), (Kids (1995),0.6000000000000001), (Before Sunrise (1995),0.8), (Nobody's Fool (1994),0.8), (Dazed and Confused (1993),0.8), (Naked (1993),0.8), (Some Folks Call It a Sling Blade (1993),0.8), (Tie Me Up! Tie Me Down! (1990),0.6000000000000001), (Stalker (1979),0.8))
      top: Vector((Back to the Future (1985),4.160090703262826), (Indiana Jones and the Last Crusade (1989),3.7648602942194134), (Fugitive, The (1993),3.7265406409612365), (E.T. the Extra-Terrestrial (1982),3.3833017514227604), (Godfather, The (1972),3.321823285277317))
      flop: Vector((Symphonie pastorale, La (1946),0.00485357562769257), (Santa Clause, The (1994),0.004797661239001768), (Mrs. Winterbourne (1996),0.004744499357035143), (He Walked by Night (1948),0.004330323336262916), (Very Natural Thing, A (1974),0.0032422381318479502))

      rated: Vector((Contact (1997),1.0), (George of the Jungle (1997),0.6000000000000001), (Full Monty, The (1997),0.8), (English Patient, The (1996),0.8), (Scream (1996),0.8), (Liar Liar (1997),0.6000000000000001), (Air Force One (1997),0.8), (In & Out (1997),0.8), (Fly Away Home (1996),0.8), (Mrs. Brown (Her Majesty, Mrs. Brown) (1997),0.6000000000000001), (Devil's Advocate, The (1997),0.6000000000000001), (Titanic (1997),1.0), (Mother (1996),0.8), (Dante's Peak (1997),0.8), (Conspiracy Theory (1997),0.6000000000000001), (Alien: Resurrection (1997),0.8), (Jackal, The (1997),0.6000000000000001), (Seven Years in Tibet (1997),0.6000000000000001), (MatchMaker, The (1997),0.8), (Scream 2 (1997),0.8))
      top: Vector((L.A. Confidential (1997),1.8203934596560891), (Evita (1996),1.6747481372310205), (Devil's Own, The (1997),1.6069074106190242), (Star Wars (1977),1.534825964399671), (Good Will Hunting (1997),1.4855529115399035))
      flop: Vector((Tigrero: A Film That Was Never Made (1994),0.0016139519738417247), (Time Tracers (1995),0.0015415064844195828), (Old Man and the Sea, The (1958),0.0012784879147284102), (Marlene Dietrich: Shadow and Light (1996) ,8.171825046501848E-4), (While You Were Sleeping (1995),1.8757478026110885E-4))

      ...

      [success] Total time: 487 s, completed 28.07.2017 11:18:53

 */
