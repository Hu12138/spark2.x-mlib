package study

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * 基于某市出租车行驶轨迹数据，使用K-Means算法进行聚类操作
  *
  * 数据集说明：
  *    CSV格式数据集
  *    - 样本数据：
  *       1,30.624806,104.136604,211846
  *    - 字段说明：
  *       - TID： 出租车的ID
  *       - Lat: 出租车载客时 维度 数据
  *       - Lon: 出租车载客时 经度 数据
  *       - Time: 记录该条数据的时间戳 ，211846表示的是21点 18分 46秒
  */
object TaxiClusteringSpark {

  def main(args: Array[String]): Unit = {

    // TODO: 1. 创建SparkSession实例对象，读取数据
    val spark = SparkSession.builder()
      .appName("TaxiClusteringSpark")
      .master("local[4]")
      .getOrCreate()

    // 获取SparkContext实例对象
    val sc = spark.sparkContext
    // 设置日志级别
    sc.setLogLevel("WARN")

    // TODO: 2. 读取出租车轨迹数据，spark.read.csv读取
    // 2.1 定义数据schema信息
    val schema: StructType = StructType(
      Array(
        StructField("tid", StringType, nullable = true),
        StructField("lat", DoubleType, nullable = true),
        StructField("lon", DoubleType, nullable = true),
        StructField("time", StringType, nullable = true)
      )
    )

    // 2.2 读取CSV格式数据
   val taxiDF: DataFrame = spark.read
      .option("header", "false")
      .schema(schema)
      .csv("src/main/data/taxi.csv")

    // 读取数据以后，查看基本信息
    println(s"Count = ${taxiDF.count()}")
    taxiDF.printSchema()
    taxiDF.show(numRows = 5, truncate = false)

    /**
      * 合并 多列数据到一个向量vector中，使用VectorAssembler
      */
    // 定义合并哪些列
    val columns = Array("lat", "lon")
    // 创建一个向量装配器VectorAssembler，设置合并列名和输出的列名
    val vectorAssembler: VectorAssembler =  new VectorAssembler()
      // 设置输入列名
      .setInputCols(columns)
      // 设置输出的列名
      .setOutputCol("features")
    // 使用转换器转换数据
    val taxiFeaturesDF: DataFrame = vectorAssembler.transform(taxiDF)

    taxiFeaturesDF.printSchema()
    taxiFeaturesDF.show(numRows = 5, truncate = false)

    // 将数据集划分为训练集和测试集
    val Array(traingDF, testingDF) = taxiFeaturesDF.randomSplit(Array(0.7, 0.3), seed = 123L)


    /**
      * 将数据使用KMeans模型学习器进行训练学习得到模型
      */
    // 创建KMeans模型学习器实例对象（算法实例对象）
    val km = new KMeans()
      .setK(2) // 设置类簇中心点个数
      .setMaxIter(20) // 设置最大的迭代次数
      .setFeaturesCol("features") // 设置模型学习器使用数据的列名称
      .setPredictionCol("prediction") // 设置模型学习器得到模型以后预测数据值的列名称
    // 使用训练数据应用到模型学习器中，训练模型
    val kmModel: KMeansModel = km.fit(traingDF)

    // 获取KMeans模型聚类中心
    val kmResult = kmModel.clusterCenters
    println(kmResult.mkString(", "))

    // 使用模型（转换器）预测测试集，各个数据所属类簇
    val predictionDF: DataFrame = kmModel.transform(testingDF)
    // 查看预测结果
    predictionDF.show(numRows = 20, truncate = false)

    // 线程休眠
   // Thread.sleep(1000000)
    // 关闭资源
    spark.stop()
  }

}
