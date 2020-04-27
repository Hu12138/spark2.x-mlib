package ml

import ml.Titanic_ml.title_mapDict
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import utils.Middle
object Titanic_ml_concat {
  def main(args: Array[String]): Unit = {
    val conf:SparkConf =new SparkConf(true)
      .setMaster("local[4]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val DF1:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/train.csv")

    val DF2 = spark.read.option("header","true").option("inferSchema","true").csv("src/main/data/testWithSurvive.csv")
    //合并两份数据
    val titanicDF = DF1.union(DF2)
   // DF.createOrReplaceGlobalTempView("trainAndTest")
//    DF.registerTempTable("trainAndTest   ")
//    DF.sqlContext.sql("select * from trainAndTest where PassengerId>892").show()

    import spark.implicits._
    val avgAge: Double = titanicDF.select($"Age").agg("Age" -> "avg").first().getDouble(0)
    val avgFare = titanicDF.select($"Fare").agg("Fare"->"avg").first().getDouble(0)
//    val middle = spark.udf.register("middle",new Middle)
    //改成中位数
   // val avgFare = titanicDF.select($"Fare").agg("Fare"->"middle").first().getDouble(0)

    val titanicRDD: RDD[FeatureDF] = titanicDF.select($"PassengerId",$"Survived",$"Pclass",$"Name",$"Sex",$"Age",$"SibSp",$"Parch",$"Ticket",$"Fare",$"Cabin",$"Embarked").rdd.map(row=>{
      var embarked: String = row.getString(11)
      if(embarked == null){
        embarked = "S"
      }
      val cabinName = row.getString(10)
      val cabin = if(cabinName == null) "U" else cabinName.substring(0,1)
      //名称头衔
      val titleName:String = title_mapDict(row.getString(3))
      //填充缺省值
      //val ages = if(row.get(5) != null) row.getDouble(5) else avgAge
      //用称谓中位数代替
      /*+-------+-----------+
|  title|middle(age)|
+-------+-----------+
|   Miss|       24.0|
| Anyone|       18.0|
|Officer|       36.0|
|Royalty|       39.0|
| Master|        3.0|
|     Mr|       25.0|
|    Mrs|       31.5|
*/
      val ages = if(row.get(5) != null) row.getDouble(5) else getMiddleAge(titleName)
      //年龄分类
      val age = if(ages >0 && ages <11) 0 else if (ages >10 && ages <21) 1
      else if (ages >20 && ages <31) 2
      else if (ages >30 && ages <41) 3
      else if (ages >40 && ages <51) 4
      else if (ages >50 && ages <61) 5
      else if (ages >60 && ages <71) 6
      else 7


      val familySize = row.getInt(6)+row.getInt(7)
      val familyCate: Int = if(familySize<2) 1 else if (familySize >=2 && familySize<=3) 2 else 3
      val passengerId = row.getInt(0)
      val survived = row.getInt(1)
      val Pclass = row.getInt(2)
      val sex = row.getString(4)
      //填充缺省fare
      val Fares = if(row.get(9) != null)row.getDouble(9) else avgFare
      //fare分类
      val Fare = if(Fares >=0 && Fares <8) 0
      else if (Fares >=8 && Fares <15) 1
      else if (Fares >=15 && Fares <32) 2
      else 3
      //      Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      //   Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      FeatureDF(passengerId,survived,titleName,Pclass,sex,age,familyCate,Fare,cabin,embarked)
    }
    )
    //初步筛选的数据集
    //所有的数据
    val allDF: DataFrame = titanicRDD.toDF()
    //注册表
    allDF.registerTempTable("allTable")
    //筛选训练数据
    val trainDF = allDF.sqlContext.sql("select * from allTable where PassengerId <892")
    //筛选测试数据
    val testDF = allDF.sqlContext.sql("select * from allTable where PassengerId >=892")
    //featureDF.show(20)
    val titleIndex = new StringIndexer().setInputCol("titleName").setOutputCol("titleNameIndex").setHandleInvalid("keep")
    val sexIndex = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex").setHandleInvalid("keep")
    val cabinIndex = new StringIndexer().setInputCol("cabin").setOutputCol("cabinIndex").setHandleInvalid("keep")
    val embarked = new StringIndexer().setInputCol("embarked").setOutputCol("embarkedIndex").setHandleInvalid("keep")

    val encoder = new OneHotEncoderEstimator().setInputCols(Array("titleNameIndex","Pclass","sexIndex","age","familyCate","Fare","cabinIndex","embarkedIndex"))
      .setOutputCols(Array("titleNameEncoded","PclassEncoded","sexEncoded","ageEncoded","familyCateEncoded","FareEncoded","cabinEncoded","embarkedEncoded"))
      .setHandleInvalid("keep")
    val vectorAssembler = new VectorAssembler().setInputCols(
      Array("titleNameEncoded","PclassEncoded","sexEncoded","ageEncoded","familyCateEncoded","FareEncoded","cabinEncoded","embarkedEncoded")
    )
      .setOutputCol("Features")
      .setHandleInvalid("keep")
    val DT = new RandomForestClassifier().setFeaturesCol("Features").setLabelCol("survived").setPredictionCol("prediction")
      .setImpurity("gini").setSeed(3)

    val pipeline = new Pipeline().setStages(Array(titleIndex,sexIndex,cabinIndex,embarked,encoder,vectorAssembler,DT))

    //再加一个交叉验证
    val paraMap = new ParamGridBuilder()
      .addGrid(DT.maxDepth,Array(4,6,8))

      .addGrid(DT.maxBins,Array(8,16,32))
      .addGrid(DT.featureSubsetStrategy,Array("auto","sqrt","log2"))
      .addGrid(DT.minInstancesPerNode,Array(1,3,10))
      .build()
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setNumFolds(5)
      .setEstimatorParamMaps(paraMap)
      .setEvaluator(
        new BinaryClassificationEvaluator()
          .setLabelCol("survived")
          .setRawPredictionCol("prediction")
          .setMetricName("areaUnderPR")
      )
    val model: CrossValidatorModel = crossValidator.fit(trainDF)

    val prediction: DataFrame = model.transform(testDF)
    prediction.select("passengerId","prediction").write.format("csv").save("src/main/data/result12.csv")
   // prediction.select("survived","titleNameEncoded","PclassEncoded","sexEncoded","familyCateEncoded","cabinEncoded","embarkedEncoded").write.format("csv").save("src/main/data/result9.csv")
    spark.stop()

  }
  def getMiddleAge(title:String): Double ={
    val middleAge = title match {
      case "Miss" =>24.0
      case "Anyone" =>18.0
      case "Officer" =>36.0
      case "Royalty" =>39.0
      case "Master" =>3.0
      case "Mr" =>25.0
      case "Mrs" =>31.0
    }
    middleAge
  }
}
