import com.atilika.kuromoji.ipadic.Tokenizer
import com.opencsv.CSVReader
import com.opencsv.bean.ColumnPositionMappingStrategy
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import java.io.FileReader
import com.opencsv.bean.CsvToBeanBuilder
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator


/**
 * Created by masaki on 2017/08/27.
 */

fun main(args: Array<String>) {
    println("Hello, World!")

    val numRows = 2
    val numColumns = 2
    val nChannels = 4
    val outputNum = 4
    val numSamples = 150
    val batchSize = 110
    val iterations = 10
    val splitTrainNum = 100
    val seed: Long = 123
    val listenerFreq = 1
    val lstmLayerSize = 200

    val maxWordNum = 40


    /**
     *Set a neural network configuration with multiple layers
     */
    println("Load data....")
    val reader = CSVReader(FileReader("data.csv"), ',', '"', 1)
    val lines = reader.readAll()
    val csvX = lines.map { line -> line[0] }
    val csvY = lines.map { line -> line[1].toFloat() }
    println(csvX)
    println(csvY)

    println("形態素解析")
    val tokenizer = Tokenizer()
    val wordToSequence = hashMapOf<String, Int>()
    var sequence = 0
    val wordX:List<List<String>> = csvX.map { x ->
        tokenizer.tokenize(x).map { attr ->
            val word = attr.getSurface()
            if (!wordToSequence.contains(word))
                wordToSequence.put(word, sequence++)
            word
        }
    }
    println(wordX.forEach { v -> println(v) })
    println(wordToSequence)

    val fullX: List<List<Float>> = wordX.map { words -> words.map { word -> wordToSequence.getOrDefault(word, 0).toFloat() } }
    println(fullX.forEach { v -> println(v) })

    println("データ整形")
    val fillZeroX: List<List<Float>> = fullX.map { words -> words + Collections.nCopies(maxWordNum - words.size, 0.0f) }
    println(fillZeroX.forEach { v -> println(v) })

    println("トレーニング＆テストに分割")
    val x: INDArray = Nd4j.create(fillZeroX.map { v -> v.toFloatArray() }.toTypedArray())
    val y: INDArray = Nd4j.create(csvY.map { v ->
        when {
            0 <= v && v < 20 -> {
                arrayOf(1.0f, 0.0f, 0.0f, 0.0f)
            } 20 <= v && v < 30 -> {
                arrayOf(0.0f, 1.0f, 0.0f, 0.0f)
            } 30 <= v && v < 40 -> {
                arrayOf(0.0f, 0.0f, 1.0f, 0.0f)
            } else -> {
                arrayOf(0.0f, 0.0f, 0.0f, 1.0f)
            }
        }.toFloatArray()
    }.toTypedArray())
    val dataSet = DataSet(x, y)
    val trainTest: SplitTestAndTrain = SplitTestAndTrain(dataSet, dataSet)


    val conf = NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, DenseLayer.Builder()
                    .nIn(maxWordNum)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build())
            .layer(1, GravesLSTM.Builder().nIn(1000).nOut(lstmLayerSize)
                    .activation(Activation.TANH).build())
            .layer(2, GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                    .activation(Activation.TANH).build())
            .layer(3, RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nIn(lstmLayerSize)
                    .nOut(outputNum)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .updater(Updater.RMSPROP)
                    .build())
            .backprop(true).pretrain(false).build()

    println("Build model....")
    val model: MultiLayerNetwork = MultiLayerNetwork(conf)
    model.init()
    model.setListeners(Arrays.asList(ScoreIterationListener(listenerFreq) as IterationListener))

    println("Train model....")
    println("Training on " + trainTest.getTrain().labelCounts())
    model.fit(trainTest.getTrain())

    println("Evaluate weights....")
    for(layer in model.getLayers()) {
        val w: INDArray = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
        println("Weights: " + w)
    }

    println("Evaluate model....")
    println("Training on " + trainTest.getTest().labelCounts())

    val eval = Evaluation(outputNum)
    val output: INDArray = model.output(trainTest.getTest().getFeatureMatrix())
    eval.eval(trainTest.getTest().getLabels(), output)
    println(eval.stats())

    println("****************Example finished********************")

//    val irisIter: BaseDatasetIterator = IrisDataSetIterator(150, 150)
//    val iris: DataSet = irisIter.next()
//    iris.normalizeZeroMeanZeroUnitVariance()
//    println("Loaded " + iris.labelCounts())
//    Nd4j.shuffle(iris.getFeatureMatrix(), Random(seed), 1)
//    Nd4j.shuffle(iris.getLabels(), Random(seed),1)
//    val trainTest: SplitTestAndTrain = iris.splitTestAndTrain(splitTrainNum, Random(seed))
//
//    val conf = NeuralNetConfiguration.Builder()
//            .seed(seed)
//            .iterations(iterations)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .list()
//            .layer(0, DenseLayer.Builder()
//                    .nIn(nChannels)
//                    .nOut(1000)
//                    .activation(Activation.RELU)
//                    .weightInit(WeightInit.RELU)
//                    .build())
//            .layer(1, GravesLSTM.Builder().nIn(1000).nOut(lstmLayerSize)
//                    .activation(Activation.TANH).build())
//            .layer(2, GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                    .activation(Activation.TANH).build())
//            .layer(3, RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                    .nIn(lstmLayerSize)
//                    .nOut(outputNum)
//                    .weightInit(WeightInit.XAVIER)
//                    .activation(Activation.SOFTMAX)
//                    .updater(Updater.RMSPROP)
//                    .build())
//            .backprop(true).pretrain(false).build()
//
//    println("Build model....")
//    val model: MultiLayerNetwork = MultiLayerNetwork(conf)
//    model.init()
//    model.setListeners(Arrays.asList(ScoreIterationListener(listenerFreq) as IterationListener))
//
//    println("Train model....")
//    println("Training on " + trainTest.getTrain().labelCounts())
//    model.fit(trainTest.getTrain())
//
//    println("Evaluate weights....")
//    for(layer in model.getLayers()) {
//        val w: INDArray = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
//        println("Weights: " + w)
//    }
//
//    println("Evaluate model....")
//    println("Training on " + trainTest.getTest().labelCounts())
//
//    val eval = Evaluation(outputNum)
//    val output: INDArray = model.output(trainTest.getTest().getFeatureMatrix())
//    eval.eval(trainTest.getTest().getLabels(), output)
//    println(eval.stats())
//
//    println("****************Example finished********************")
}