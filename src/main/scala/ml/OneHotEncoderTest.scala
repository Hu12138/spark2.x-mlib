package ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, OneHotEncoderModel, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost, spark}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}

import scala.collection.mutable.ListBuffer

object OneHotEncoderTest {
  def main(args: Array[String]): Unit = {
    // TODO: 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("TitanicClassificationMLTest")
      .master("local[4]")
      .getOrCreate()
    // 导入隐式转换
    import spark.implicits._

    // 获取SparkContext实例对象
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    /**
     * TODO: a. 读取泰坦尼克号数据集
     */
    val titanicDF: DataFrame = spark.read
      .option("header", "true").option("inferSchema", "true")
      .csv("src/main/data/train.csv")
    //val titanicDF = RdsConn.TrainDataFrame("titanic")
    // 样本数据
    titanicDF.show(10, truncate = false)
    titanicDF.printSchema()
    val dataDF = titanicDF.select(
      $"Survived", $"Pclass", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare"
    )
//    val categoricalColumns: Array[String] = Array("Sex")
//    val stagesArray: ListBuffer[Pipeline] = new ListBuffer[Pipeline]()
//    for(cate <- categoricalColumns){
//      val indexer = new StringIndexer().setInputCol(cate).setOutputCol(s"${cate}Index")
//
//      val encoder = new OneHotEncoder().setInputCol(indexer.getOutputCol).setOutputCol(s"${cate}classVector")
//      stagesArray.append(indexer,encoder)
//      encoder.fit()
//    }
    val categoricals: Array[String] = dataDF.dtypes.filter(_._2 == "StringType").map (_._1)
   // categoricals.map(x=>println(x))
    val indexers: Array[StringIndexer] = categoricals.map(
      c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_idx").setHandleInvalid("keep")
    )
    //indexers.map(x=>println(x))
//    val encoders: Array[OneHotEncoderEstimator] = categoricals.map(
//      c => new OneHotEncoderEstimator().setInputCols(Array(s"${c}_idx")).setOutputCols(Array(s"${c}_enc")).setDropLast(false).setHandleInvalid("keep")
//    )
    val encoders = new OneHotEncoderEstimator().setInputCols(categoricals).setOutputCols(categoricals.map(x=>s"${x}_enc"))
    val b: OneHotEncoderModel = encoders.fit(dataDF)
    val c: DataFrame = b.transform(dataDF)
    c.show(10)

spark.stop()
  }
}