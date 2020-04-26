package utils

import ml.FeatureDF
import ml.Titanic_ml.title_mapDict
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

object Test {
  def main(args: Array[String]): Unit = {
    val conf:SparkConf =new SparkConf(true)
      .setMaster("local[4]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val testDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/test.csv")
    testDF.printSchema()

    import spark.implicits._
    val avgAge: Double = testDF.select($"Age").agg("Age" -> "avg").first().getDouble(0)

    val titanicRDD: RDD[FeatureDF] = testDF
      .select($"PassengerId",$"Pclass",$"Name",$"Sex",$"Age",$"SibSp",$"Parch",$"Ticket",$"Fare",$"Cabin",$"Embarked")
      .rdd.map(row=>{
      var embarked: String = row.getString(10)
      if(embarked == null){
        embarked = "S"
      }
      var cabin = row.getString(9)
      if(cabin == null){
        cabin = "Unknown"
      }
      val age = if(row.get(4) != null) row.getDouble(4) else avgAge

      //名称头衔
      var titleName:String = title_mapDict(row.getString(2))

      val familySize = row.getInt(5)+row.getInt(6)
      var familyCate: Int = if(familySize<2) 1 else if (familySize >=2 && familySize<=4) 2 else 3
      val passengerId = row.getInt(0)
      //val survived = row.getInt(1)
      val Pclass = row.getInt(1)
      var sex = row.getString(3)
      val Fare = row.getDouble(8)
      //      Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      //   Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      FeatureDF(passengerId,0,titleName,Pclass,sex,age,familyCate,Fare,cabin,embarked)
    }
    )
    //初步筛选的数据集
    val featureDF: DataFrame = titanicRDD.toDF()
    featureDF.show(10)
    spark.stop()
  }

}
