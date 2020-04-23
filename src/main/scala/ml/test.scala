package ml
import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LinearSVC, LinearSVCModel, LogisticRegression}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, OneHotEncoderModel, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object test {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf(true).setMaster("local[2]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val titanicDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(args(0))

    val ttdf: DataFrame = titanicDF.select("Survived","Sex","Age","SibSp","Parch","Fare")
    val Array(trainDF,testDF) = ttdf.randomSplit(Array(0.7,0.3))
    val sexIndex: StringIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val sexIndexModel: StringIndexerModel = sexIndex.fit(ttdf)
    val sexIndexed: DataFrame = sexIndexModel.transform(ttdf)
    val sexEncode: OneHotEncoderEstimator = new OneHotEncoderEstimator().setInputCols(Array("SexIndex")).setOutputCols(Array("SexEncode"))
    val sexEncodeModel: OneHotEncoderModel = sexEncode.fit(sexIndexed)
    val sexEncoded = sexEncodeModel.transform(sexIndexed)
    println("====================================")
    sexIndexed.show(10)
    println("====================================")
    sexEncoded.show(10)
    val vector: VectorAssembler = new VectorAssembler().setInputCols(Array("SexEncode","SibSp","Parch","Fare")).setOutputCol("features")
    val vectorAssemblered: DataFrame = vector.transform(sexEncoded)

    println("==========================================")
    vectorAssemblered.show(10)
     val Array(trainData,testData) = vectorAssemblered.randomSplit(Array(0.7,0.3))
    println("==========================================")
        val svm = new DecisionTreeClassifier()
            .setLabelCol("Survived")
            .setFeaturesCol("features")
            .setPredictionCol("prediction")
    val svmModel: DecisionTreeClassificationModel = svm.fit(trainData)
    val svmDF = svmModel.transform(testData)
    svmDF.show(10)
    spark.stop()
  }
}