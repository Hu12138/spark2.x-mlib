package utils

import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.sql.{DataFrame, SparkSession}

object RdsConn {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().master("local").appName("RdsConn")
      .config("spark.sql.shuffle.partitions", 1).getOrCreate()
    val url = "jdbc:mysql://rm-bp133k9z9m409gthbro.mysql.rds.aliyuncs.com:3306/hxf" + "?autoReconnect=true&useUnicode=true&useSSL=false" + "&characterEncoding=utf-8&serverTimezone=UTC"
    val userName = "root"
    val password = "root1234"
    //val database = "hxf"
    val map: Map[String, String] = Map[String, String](
      elems = "url" -> url,
      "driver" -> "com.mysql.jdbc.Driver",
      "user" -> userName,
      "password" -> password,
      "dbtable"->"rtzb"
    )

    val rtzb: DataFrame = spark.read.format("jdbc").options(map).load
//    val map2: Map[String, String] = Map[String, String](
//      elems = "url" -> url,
//      "driver" -> "com.mysql.jdbc.Driver",
//      "user" -> userName,
//      "password" -> password,
//      "dbtable"->"titanic"
//    )
    rtzb.show()
    rtzb.printSchema()
//    val titanic: DataFrame = spark.read.format("jdbc").options(map2).load
//    titanic.show(10)
val titanicDF: DataFrame = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv("src/main/data/train.csv")
    //val titanicDF = RdsConn.TrainDataFrame("titanic")
    // 样本数据
    titanicDF.show(10, truncate = false)
    titanicDF.printSchema()
  }
  def TrainDataFrame(dataTable:String): DataFrame ={
    val spark: SparkSession = SparkSession.builder().master("local").appName("RdsConn")
      .config("spark.sql.shuffle.partitions", 1).getOrCreate()
    val url = "jdbc:mysql://rm-bp133k9z9m409gthbro.mysql.rds.aliyuncs.com:3306/hxf" + "?autoReconnect=true&useUnicode=true&useSSL=false" + "&characterEncoding=utf-8&serverTimezone=UTC"
    val userName = "root"
    val password = "root1234"
    //val database = "hxf"
    val map: Map[String, String] = Map[String, String](
      elems = "url" -> url,
      "driver" -> "com.mysql.jdbc.Driver",
      "user" -> userName,
      "password" -> password,
      "dbtable"->dataTable
    )
    val dataTableDataFrame = spark.read.format("jdbc").options(map).load
    //val encodeHot = new OneHotEncoderEstimator().set
    dataTableDataFrame
  }
}
