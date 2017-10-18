import com.atilika.kuromoji.ipadic.Tokenizer
import com.opencsv.CSVReader
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import java.io.FileReader
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator

fun main(args: Array<String>) {
    val outputNum = 6
    val iterations = 10
    val seed: Long = 123
    val listenerFreq = 20
    val lstmLayerSize = 200
    val batchSize = 64
    val numEpochs = 10

    val maxWordNum = 1000

    println("===== データ取得 =====")
    val reader = CSVReader(FileReader("data.csv"), ',', '"', 1)
    val lines = reader.readAll()
    val csvX = lines.map { line -> line[0] }
    val csvY = lines.map { line -> line[2].toFloat() }
    println(csvX)
    println(csvY)

    println("===== 形態素解析 =====")
    val tokenizer = Tokenizer()
    val wordSequenceMap = hashMapOf<String, Int>()
    var sequence = 0
    val wordX = csvX.map { x ->
        tokenizer.tokenize(x).map { info ->
            val word = info.getSurface()
            if (!wordSequenceMap.contains(word))
                wordSequenceMap.put(word, sequence++)
            word
        }
    }
    println(wordX.forEach { v -> println(v) })
    println(wordSequenceMap)

    val sequencesX = wordX.map { x -> x.map { word -> wordSequenceMap.getOrDefault(word, 0).toFloat() } }
    println(sequencesX.forEach { v -> println(v) })

    println("===== データ整形 =====")
    val fillZeroX = sequencesX.map { x -> x + Collections.nCopies(maxWordNum - x.size, 0.0f) }
    println(fillZeroX.forEach { v -> println(v) })

    println("===== トレーニング＆テストに分割 =====")
    val x: INDArray = Nd4j.create(fillZeroX.map { v -> v.toFloatArray() }.toTypedArray())
    val y: INDArray = Nd4j.create(csvY.map { v ->
        when {
            0 <= v && v < 20 -> {
                arrayOf(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
            } 20 <= v && v < 30 -> {
                // 20代
                arrayOf(0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f)
            } 30 <= v && v < 40 -> {
                // 30代
                arrayOf(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f)
            } 40 <= v && v < 50 -> {
                // 40代
                arrayOf(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f)
            } 50 <= v && v < 60 -> {
                // 50代
                arrayOf(0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f)
            } else -> {
                // 60以上
                arrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)
            }
        }.toFloatArray()
    }.toTypedArray())
    val dataSet = DataSet(x, y)
    dataSet.shuffle()
    val trainTest: SplitTestAndTrain = dataSet.splitTestAndTrain(0.8)

    println("===== モデル作成 =====")
    val conf = NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, GravesLSTM.Builder()
                    .nIn(maxWordNum)
                    .nOut(lstmLayerSize)
                    .activation(Activation.TANH)
                    .build())
            .layer(1, RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nIn(lstmLayerSize)
                    .nOut(outputNum)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .updater(Updater.RMSPROP)
                    .build())
            .backprop(true).pretrain(false).build()

    val model: MultiLayerNetwork = MultiLayerNetwork(conf)
    model.init()
    model.setListeners(Arrays.asList(ScoreIterationListener(listenerFreq) as IterationListener))

    println("===== 学習 =====")
    for (i in 0..numEpochs - 1) {
        model.fit(trainTest.getTrain())

        val eval = model.evaluate(ListDataSetIterator(trainTest.getTrain().asList(), trainTest.getTrain().asList().size))
        println(String.format("Epoch %d: Accuracy = %.2f, F1 = %.2f", i, eval.accuracy(), eval.f1()))
    }

    println("===== 評価 =====")
    val eval = Evaluation(outputNum)
    for (testDataSet in trainTest.test.asList()) {
        val output: INDArray = model.output(testDataSet.getFeatureMatrix())
        eval.eval(testDataSet.labels, output)
        println("▼▼▼▼▼ メモ ▼▼▼▼▼")
        println("▲▲▲▲▲ メモ ▲▲▲▲▲")
        println("テストデータ=" + testDataSet.labels + " 精度=" + output)
    }

    println(eval.stats())
}