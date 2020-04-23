package ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

object PipelineTest {
  def main(args: Array[String]): Unit = {
    val conf:SparkConf =new SparkConf(true)
      .setMaster("local[4]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val titanicDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/train.csv")
    //
    titanicDF.registerTempTable("titanicDFview")
    val avgAge: Double = titanicDF.select("Age").agg("Age"->"avg").first().getDouble(0)
    val DF1 = titanicDF.sqlContext.sql(s"select nvl(Age,$avgAge) from titanicDFview")
    val Array(trainDF,testDF) = DF1.randomSplit(Array(0.8,0.3))
    val sexIndex: StringIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")

    val sexEncode: OneHotEncoderEstimator = new OneHotEncoderEstimator().setInputCols(Array("SexIndex")).setOutputCols(Array("SexEncode"))
    val vector: VectorAssembler = new VectorAssembler().setInputCols(Array("SexEncode","SibSp","Parch","Fare")).setOutputCol("features")
    val DT = new DecisionTreeClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    val pipeline = new Pipeline().setStages(Array(sexIndex,sexEncode,vector,DT))
//    val model: PipelineModel = pipeline.fit(trainDF)
    //val prediction = model.transform(testDF)
//    prediction.show(10)
    val paramMap: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(DT.maxDepth,Array(6,8,10))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setNumFolds(3)
      .setEstimatorParamMaps(paramMap)
      .setEvaluator(new BinaryClassificationEvaluator()
          .setLabelCol("Survived")
        .setRawPredictionCol("prediction")
        .setMetricName("areaUnderROC")
    )

    val model = crossValidator.fit(trainDF)

    model.transform(testDF).select("Survived","Sex","Age","probability","prediction").show()
  }
}
