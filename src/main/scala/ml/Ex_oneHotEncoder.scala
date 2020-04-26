package ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

//import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
object Ex_oneHotEncoder {
  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf(true).setMaster("local[4]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val titanicDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/train.csv")

    val ttdf: DataFrame = titanicDF.select("Survived","Sex","Age","SibSp","Parch","Fare")
    val Array(trainDF,testDF) = ttdf.randomSplit(Array(0.7,0.3))

    val sexIndex = new StringIndexer().setInputCol("Sex").setOutputCol("sexIndex")

    println("--------------------------------------")
    //val df: DataFrame = model1.transform(titanicDF)
    val encoder = new OneHotEncoderEstimator().setInputCols(Array("sexIndex"))
        .setOutputCols(Array("SexEncoder"))
        .setDropLast(false)
      .setHandleInvalid("keep")

    val vectorAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("Survived","Age","SibSp","Parch","Fare","SexEncoder"))
      .setOutputCol("features")
      .setHandleInvalid("keep")


    val lr= new LogisticRegression().setLabelCol("Survived").setFeaturesCol("features").setPredictionCol("prediction")
    val pipeline = new Pipeline()
    pipeline.setStages(Array(sexIndex,encoder,vectorAssembler,lr))
    val model: PipelineModel = pipeline.fit(trainDF)
    val result: DataFrame = model.transform(testDF)
    result.show(10)

    spark.stop()
  }
}
