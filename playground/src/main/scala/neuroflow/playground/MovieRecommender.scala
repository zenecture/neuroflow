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

  val all = movies.map(m => ratings.count(_.movieId == m.id))
  val avg = all.sum.toDouble / movies.size.toDouble


  println("avg = " + avg)
  println(s"tts: $tts, fls: $fls")

  val xs = ratings.groupBy(_.user).map {
    case (userId, rs) => rs.map { r =>
      ζ(movies.size).updated(r.movieId - 1, r.rating.toDouble * 0.2)
    }.reduce(_ + _)
  }.toList

  val layout =
    Input(movies.size)            ::
    Hidden(50, Sigmoid)           ::
    Hidden(50, Sigmoid)           ::
    Hidden(50, Sigmoid)           ::
    Output(movies.size, Sigmoid)  :: HNil

  import neuroflow.nets.AutoEncoder._

  def apply = {

    import neuroflow.core.FFN.WeightProvider._

    val net = Network(layout,
      Settings(iterations = 500,
        learningRate = {
          case iter if iter < 125  =>  0.5
          case _                   =>  0.5
        }, precision = 1E-3, parallelism = 64))

    net.train(xs)

    IO.File.write(net, file)

  }

  def eval = {

    implicit val wp = IO.File.read(file)
    val net = Network(layout)
    import neuroflow.application.processor.Extensions.VectorOps

    xs.foreach { x =>
      val z = x.zipWithIndex.flatMap {
        case (i, k) if i == 0 => Some(k)
        case _ => None
      }

      val o = x.zipWithIndex.flatMap {
        case (i, k) if i > 0 => Some(movies.find(_.id == k + 1).get.title -> (i / 0.2))
        case _ => None
      }

      val r = net.evaluate(x).map(_ / 0.2)
      val all = z.map { i => i -> r(i) }.sortBy(_._2).reverse.map {
        case (i, rat) => movies.find(_.id == i + 1).get.title -> rat
      }

      val (top, flop) = (all.take(10), all.takeRight(5))

      val delta = x.map(i => i / 0.2) - r
      println("err: " + delta.sum / delta.size)

      println(s"rated: $o")
      println(s"top: $top")
      println(s"flop: $flop")
      println()
    }

  }

}

/*

        [run-main-0] INFO neuroflow.nets.DefaultNetwork - [02.08.2017 20:16:54:074] Taking step 497 - Mean Error 11,4661 - Error Vector 61.54203222163417  18.54826201486953  18.14292935512043  ... (1682 total)
        [run-main-0] INFO neuroflow.nets.DefaultNetwork - [02.08.2017 20:16:55:419] Taking step 498 - Mean Error 11,4016 - Error Vector 61.04398407183125  18.582731955568562  18.078676877228467  ... (1682 total)
        [run-main-0] INFO neuroflow.nets.DefaultNetwork - [02.08.2017 20:16:56:631] Taking step 499 - Mean Error 11,4639 - Error Vector 60.368374337678986  18.554694531357313  18.143068683116077  ... (1682 total)
        [run-main-0] INFO neuroflow.nets.DefaultNetwork - [02.08.2017 20:16:57:947] Took 500 iterations of 500 with Mean Error = 11,4235
        [success] Total time: 614 s, completed 02.08.2017 20:16:59
        > neuroflow-playground/run
        [info] Compiling 1 Scala source to /Users/felix/github/neuroflow/playground/target/scala-2.12/classes...
        [info] Running neuroflow.playground.App
        Run example (1-17): avg = 59.45303210463734
        tts: 21201, fls: 6110




                     _   __                      ________
                    / | / /__  __  ___________  / ____/ /___ _      __
                   /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
                  / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
                 /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/


                 Version 0.802

                 Identifier: N1
                 Network: neuroflow.nets.DefaultNetwork
                 Layout: [1682 In, 50 H (σ), 50 H (σ), 50 H (σ), 1682 Out (σ)]
                 Number of Weights: 173200




        Aug 02, 2017 8:18:25 PM com.github.fommil.jni.JniLoader liberalLoad
        INFORMATION: successfully loaded /var/folders/t_/plj660gn6ps0546vj6xtx92m0000gn/T/jniloader7398041489079473210netlib-native_system-osx-x86_64.jnilib
        err: -0.05668755925016253
        rated: Vector((Twelve Monkeys (1995),4.0), (Dead Man Walking (1995),4.0), (Mighty Aphrodite (1995),4.0), (Mr. Holland's Opus (1995),4.0), (Fargo (1996),4.0), (Independence Day (ID4) (1996),4.0), (Lone Star (1996),3.0000000000000004), (Spitfire Grill, The (1996),3.0000000000000004), (Bound (1996),3.0000000000000004), (Big Night (1996),3.0000000000000004), (Swingers (1996),3.0000000000000004), (Jerry Maguire (1996),4.0), (Devil's Own, The (1997),3.0000000000000004), (Contact (1997),4.0), (Chasing Amy (1997),3.0000000000000004), (Good Will Hunting (1997),5.0), (Leaving Las Vegas (1995),4.0), (Secrets & Lies (1996),3.0000000000000004), (Scream (1996),3.0000000000000004), (Liar Liar (1997),4.0), (Air Force One (1997),3.0000000000000004), (In & Out (1997),3.0000000000000004), (L.A. Confidential (1997),4.0), (Devil's Advocate, The (1997),4.0), (Titanic (1997),5.0), (Apt Pupil (1998),5.0), (Lost Highway (1997),3.0000000000000004), (G.I. Jane (1997),4.0), (Conspiracy Theory (1997),4.0), (Edge, The (1997),4.0), (Game, The (1997),4.0), (Boogie Nights (1997),5.0), (Prophecy II, The (1998),3.0000000000000004), (Wedding Singer, The (1998),3.0000000000000004), (Spawn (1997),2.0), (People vs. Larry Flynt, The (1996),4.0), (Mouse Hunt (1997),3.0000000000000004), (Seven Years in Tibet (1997),4.0), (Anne Frank Remembered (1995),3.0000000000000004))
        top: Vector((English Patient, The (1996),2.5033450581190957), (Saint, The (1997),2.154344471656136), (As Good As It Gets (1997),1.7720676189201796), (Kiss the Girls (1997),1.5748194540476206), (Full Monty, The (1997),1.5670269880222265), (Murder at 1600 (1997),1.4625601223540459), (Wag the Dog (1997),1.2281621735202393), (Alien: Resurrection (1997),1.1224147180070017), (Evita (1996),1.1211561854382086), (Rainmaker, The (1997),1.0870510270320923))
        flop: Vector((3 Ninjas: High Noon At Mega Mountain (1998),0.004025915195528413), (Favor, The (1994),0.0038095451140025016), (Even Cowgirls Get the Blues (1993),0.0035178949041584), (Shopping (1994),0.0029738968331901173), (Wedding Bell Blues (1996),0.0027897831595613985))

        err: -0.0022319267428319695
        rated: Vector((Toy Story (1995),5.0), (Muppet Treasure Island (1996),4.0), (Braveheart (1995),3.0000000000000004), (Rumble in the Bronx (1995),4.0), (Apollo 13 (1995),3.0000000000000004), (Star Wars (1977),5.0), (Pulp Fiction (1994),4.0), (Stargate (1994),3.0000000000000004), (Shawshank Redemption, The (1994),4.0), (Four Weddings and a Funeral (1994),4.0), (Lion King, The (1994),5.0), (Mask, The (1994),3.0000000000000004), (Hudsucker Proxy, The (1994),5.0), (Jurassic Park (1993),4.0), (Much Ado About Nothing (1993),5.0), (Blade Runner (1982),5.0), (Nightmare Before Christmas, The (1993),4.0), (Home Alone (1990),3.0000000000000004), (Snow White and the Seven Dwarfs (1937),4.0), (Fargo (1996),3.0000000000000004), (Heavy Metal (1981),3.0000000000000004), (Mystery Science Theater 3000: The Movie (1996),5.0), (Wallace & Gromit: The Best of Aardman Animation (1996),5.0), (Independence Day (ID4) (1996),1.0), (Wizard of Oz, The (1939),4.0), (2001: A Space Odyssey (1968),4.0), (Sound of Music, The (1965),3.0000000000000004), (Lawnmower Man, The (1992),4.0), (Fish Called Wanda, A (1988),5.0), (Monty Python's Life of Brian (1979),5.0), (Top Gun (1986),3.0000000000000004), (Return of the Pink Panther, The (1974),3.0000000000000004), (Abyss, The (1989),4.0), (Monty Python and the Holy Grail (1974),5.0), (Wrong Trousers, The (1993),5.0), (Empire Strikes Back, The (1980),5.0), (Princess Bride, The (1987),5.0), (Raiders of the Lost Ark (1981),4.0), (Brazil (1985),4.0), (12 Angry Men (1957),3.0000000000000004), (Return of the Jedi (1983),5.0), (Alien (1979),3.0000000000000004), (Psycho (1960),3.0000000000000004), (Blues Brothers, The (1980),5.0), (Grand Day Out, A (1992),4.0), (Right Stuff, The (1983),3.0000000000000004), (Terminator, The (1984),3.0000000000000004), (Dead Poets Society (1989),3.0000000000000004), (Graduate, The (1967),4.0), (Shining, The (1980),3.0000000000000004), (Back to the Future (1985),3.0000000000000004), (This Is Spinal Tap (1984),5.0), (Indiana Jones and the Last Crusade (1989),2.0), (Pink Floyd - The Wall (1982),5.0), (Field of Dreams (1989),4.0), (Star Trek: First Contact (1996),4.0), (Star Trek VI: The Undiscovered Country (1991),5.0), (Star Trek: The Wrath of Khan (1982),5.0), (Star Trek III: The Search for Spock (1984),5.0), (Star Trek IV: The Voyage Home (1986),5.0), (Sneakers (1992),3.0000000000000004), (Men in Black (1997),5.0), (Contact (1997),5.0), (Hunt for Red October, The (1990),3.0000000000000004), (Full Monty, The (1997),4.0), (English Patient, The (1996),3.0000000000000004), (Titanic (1997),5.0), (Star Trek: Generations (1994),3.0000000000000004), (Mrs. Doubtfire (1993),3.0000000000000004), (Robin Hood: Men in Tights (1993),3.0000000000000004), (Brady Bunch Movie, The (1995),1.0), (Ghost (1990),2.0), (Batman (1989),3.0000000000000004), (Close Shave, A (1995),5.0), (Mary Poppins (1964),4.0), (E.T. the Extra-Terrestrial (1982),5.0), (To Kill a Mockingbird (1962),3.0000000000000004), (Harold and Maude (1971),4.0), (Highlander (1986),4.0), (Heathers (1989),5.0), (Star Trek: The Motion Picture (1979),4.0), (Star Trek V: The Final Frontier (1989),2.0), (Like Water For Chocolate (Como agua para chocolate) (1992),4.0), (Secret of Roan Inish, The (1994),5.0), (Dragonheart (1996),3.0000000000000004), (Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963),5.0), (Casablanca (1942),5.0), (Dial M for Murder (1954),3.0000000000000004), (My Left Foot (1989),3.0000000000000004), (Lawrence of Arabia (1962),5.0), (Boot, Das (1981),4.0), (Gandhi (1982),5.0), (My Life as a Dog (Mitt liv som hund) (1985),4.0), (Englishman Who Went Up a Hill, But Came Down a Mountain, The (1995),4.0), (Beauty and the Beast (1991),4.0), (Crying Game, The (1992),3.0000000000000004), (Microcosmos: Le peuple de l'herbe (1996),3.0000000000000004), (Stand by Me (1986),4.0), (Fried Green Tomatoes (1991),3.0000000000000004), (McHale's Navy (1997),1.0), (Sex, Lies, and Videotape (1989),4.0), (Shadowlands (1993),3.0000000000000004), (Pretty Woman (1990),2.0), (Real Genius (1985),4.0), (Escape from L.A. (1996),1.0), (How to Make an American Quilt (1995),4.0), (Fast, Cheap & Out of Control (1997),4.0), (Grumpier Old Men (1995),4.0), (Koyaanisqatsi (1983),3.0000000000000004), (Tank Girl (1995),4.0), (Road to Wellville, The (1994),2.0), (Barbarella (1968),4.0))
        top: Vector((Silence of the Lambs, The (1991),2.8648903614864545), (Terminator 2: Judgment Day (1991),2.5595416779076228), (Fugitive, The (1993),2.55345856872967), (Usual Suspects, The (1995),2.3586680450247), (Aliens (1986),2.2156544130461553), (Amadeus (1984),2.1636834147130024), (Sting, The (1973),2.1616167798688903), (One Flew Over the Cuckoo's Nest (1975),2.0760814611266643), (Schindler's List (1993),2.021607822493882), (Raising Arizona (1987),2.015130420369897))
        flop: Vector((Falling in Love Again (1980),0.014074354440393603), (Shopping (1994),0.01268582602828281), (Babysitter, The (1995),0.012036390925291558), (Amityville 3-D (1983),0.011790437740538522), (Blown Away (1994),0.010632114642916685))

        err: -0.07691281973282317
        rated: Vector((Remains of the Day, The (1993),4.0), (Searching for Bobby Fischer (1993),5.0), (Godfather, The (1972),2.0), (Clockwork Orange, A (1971),1.0), (Apocalypse Now (1979),1.0), (Amadeus (1984),3.0000000000000004), (Graduate, The (1967),5.0), (Bridge on the River Kwai, The (1957),5.0), (Chasing Amy (1997),5.0), (Chasing Amy (1997),5.0), (Full Monty, The (1997),3.0000000000000004), (English Patient, The (1996),4.0), (In the Name of the Father (1993),5.0), (Schindler's List (1993),5.0), (Adventures of Priscilla, Queen of the Desert, The (1994),5.0), (E.T. the Extra-Terrestrial (1982),4.0), (To Kill a Mockingbird (1962),4.0), (Lawrence of Arabia (1962),4.0), (Boot, Das (1981),4.0), (Gandhi (1982),5.0), (Killing Fields, The (1984),5.0), (Crying Game, The (1992),5.0), (Paris Is Burning (1990),3.0000000000000004), (Philadelphia (1993),4.0), (Garden of Finzi-Contini, The (Giardino dei Finzi-Contini, Il) (1970),2.0))
        top: Vector((Contact (1997),2.5614297956832557), (Star Wars (1977),2.227385389696649), (Titanic (1997),2.1385552512588553), (Air Force One (1997),1.8080112099394148), (Fargo (1996),1.5875079962208365), (Liar Liar (1997),1.575430235880188), (Dead Man Walking (1995),1.5350297678143514), (Saint, The (1997),1.3727031815407413), (In & Out (1997),1.3156781440450973), (Good Will Hunting (1997),1.3079122101593126))
        flop: Vector((Nowhere (1997),0.004336963559055207), (Girls Town (1996),0.00421845010071533), (Captives (1994),0.004153117046437132), (Wedding Bell Blues (1996),0.003884896309795944), (Even Cowgirls Get the Blues (1993),0.002848635836347112))

        ...


 */
