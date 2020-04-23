package utils

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object RenBaoFilter {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf(true).setMaster("local[2]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val rbDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(args(0))
    //rbDF.show(10)

  }

}
