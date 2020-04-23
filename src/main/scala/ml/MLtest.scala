package ml
import org.apache.spark
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object MLtest {
  def main(args: Array[String]): Unit = {
    //spark.read.format("libsvm").load()
    val spark = SparkSession.builder().appName("MLtest").master("local[4]").getOrCreate()
    //spark.read.format
    import spark.implicits._
    val sc = spark.sparkContext
    sc.setLogLevel("warn")

    val titanicDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/train.csv")

    titanicDF.show(10,truncate = false)
    titanicDF.printSchema()

    val avgAge = titanicDF.select("Age").agg("Age"->"avg").first().getDouble(0)

    val titanicRDD = titanicDF.select(
      $"Survived",$"Pclass",$"Sex",$"Age",$"SibSp",$"Parch",$"Fare"
    ).rdd.map(row=>{
      val label = row.getInt(0).toDouble

      // TODO: 针对Sex特征进行处理：把Sex变量的取值male替换为1，female替换为0
      val sexFeature = if("male".equals(row.getString(2))) 1.0 else 0.0
      //val sexFeature2 = new OneHotEncoderEstimator().setInputCols(new Array[String](""))

      // TODO: 针对Age特征进行转换：有117个乘客年龄值有缺失，用平均年龄30岁替换
      val ageFeature = if(row.get(3) == null) avgAge else row.getDouble(3)

      // 获取特征值
      val features = Vectors.dense(
        Array(row.getInt(1).toDouble, sexFeature, ageFeature,
          row.getInt(4).toDouble, row.getInt(5).toDouble, row.getDouble(6)
        )
      )
      // 返回标签向量
      LabeledPoint(label, features)
    }
    )

    val Array(trainRDD, testRDD) = titanicRDD.randomSplit(Array(0.8, 0.2))

    /**
     * TODO：c.  使用二分类算法训练模型：SVM、LR、DT和RF、GBT
     */
    // TODO: c.1. 支持向量机
//    val svmModel: SVMModel = SVMWithSGD.train(trainRDD, 100)
//    val svmPredictionAndLabels: RDD[(Double, Double)] = testRDD.map{
//      case LabeledPoint(label, features) => (svmModel.predict(features), label)
//    }
//    val svmMetrics = new BinaryClassificationMetrics(svmPredictionAndLabels)
//    println(s"使用SVM预测评估ROC: ${svmMetrics.areaUnderROC()}")
//    // TODO: c.2. 逻辑回归
//    val lrModel: LogisticRegressionModel = LogisticRegressionWithSGD.train(trainRDD, 100)
//    val lrPredictionAndLabels: RDD[(Double, Double)] = testRDD.map{
//      case LabeledPoint(label, features) => (lrModel.predict(features), label)
//    }
//    val lrMetrics = new BinaryClassificationMetrics(lrPredictionAndLabels)
//    println(s"使用LogisticRegression预测评估ROC: ${lrMetrics.areaUnderROC()}")

  }

}
