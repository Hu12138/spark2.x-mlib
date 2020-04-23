package ml

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object DataTest {
  def main(args: Array[String]): Unit = {
    val conf:SparkConf =new SparkConf(true)
      .setMaster("local[4]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val titanicDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/train.csv")
    //
    val avgAge: Double = titanicDF.select("Age").agg("Age"->"avg").first().getDouble(0)
    println(avgAge)
    titanicDF.registerTempTable("titanicDFview")
    val DF1 = titanicDF.sqlContext.sql(s"select *,nvl(Age,$avgAge) as avgAge from titanicDFview")
    DF1.show()
    spark.stop()
  }
}
