package ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

object Titanic_ml {
  def main(args: Array[String]): Unit = {

//连接数据源
    val conf:SparkConf =new SparkConf(true)
          .setMaster("local[4]").setAppName("spark ml")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val titanicDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/train.csv")
      //打印schema
    titanicDF.printSchema()
    titanicDF.na.fill(0)
    //清洗字段
    //算出age平均值
    import spark.implicits._
   val avgAge: Double = titanicDF.select($"Age").agg("Age" -> "avg").first().getDouble(0)

    val titanicRDD: RDD[FeatureDF] = titanicDF.select($"PassengerId",$"Survived",$"Pclass",$"Name",$"Sex",$"Age",$"SibSp",$"Parch",$"Ticket",$"Fare",$"Cabin",$"Embarked").rdd.map(row=>{
      var embarked: String = row.getString(11)
      if(embarked == null){
        embarked = "S"
      }
      var cabin = row.getString(10)
      if(cabin == null){
        cabin = "Unknown"
      }
      val age = try {if(row.getDouble(5) != null) row.getDouble(5) else avgAge} finally avgAge

       //名称头衔
      var titleName:String = title_mapDict(row.getString(3))

      val familySize = row.getInt(6)+row.getInt(7)
      var familyCate: Int = if(familySize<2) 1 else if (familySize >=2 && familySize<=4) 2 else 3
      val passengerId = row.getInt(0)
      val survived = row.getInt(1)
      val Pclass = row.getInt(2)
      var sex = row.getString(4)
      val Fare = row.getDouble(9)
//      Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
   //   Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      FeatureDF(passengerId,survived,titleName,Pclass,sex,age,familyCate,Fare,cabin,embarked)
    }
    )
    //初步筛选的数据集
    val featureDF: DataFrame = titanicRDD.toDF()
    //featureDF.show(20)
    val titleIndex = new StringIndexer().setInputCol("titleName").setOutputCol("titleNameIndex").setHandleInvalid("keep")
    val sexIndex = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex").setHandleInvalid("keep")
    val cabinIndex = new StringIndexer().setInputCol("cabin").setOutputCol("cabinIndex").setHandleInvalid("keep")
    val embarked = new StringIndexer().setInputCol("embarked").setOutputCol("embarkedIndex").setHandleInvalid("keep")

    val encoder = new OneHotEncoderEstimator().setInputCols(Array("titleNameIndex","Pclass","sexIndex","familyCate","cabinIndex","embarkedIndex"))
        .setOutputCols(Array("titleNameEncoded","PclassEncoded","sexEncoded","familyCateEncoded","cabinEncoded","embarkedEncoded"))
      .setHandleInvalid("keep")
    val vectorAssembler = new VectorAssembler().setInputCols(Array("survived","titleNameEncoded","PclassEncoded","sexEncoded","familyCateEncoded","cabinEncoded","embarkedEncoded"))
        .setOutputCol("Features")
      .setHandleInvalid("keep")
    val DT = new RandomForestClassifier().setFeaturesCol("Features").setLabelCol("survived").setPredictionCol("prediction")
        .setFeatureSubsetStrategy("auto").setImpurity("gini")

    val pipeline = new Pipeline().setStages(Array(titleIndex,sexIndex,cabinIndex,embarked,encoder,vectorAssembler,DT))

    //再加一个交叉验证
    val paraMap = new ParamGridBuilder()
      .addGrid(DT.maxDepth,Array(6,8,10))
      .addGrid(DT.numTrees,Array(3,4,5))
      .addGrid(DT.maxBins,Array(8,16,32))
      .build()
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setNumFolds(3)
      .setEstimatorParamMaps(paraMap)
      .setEvaluator(
        new BinaryClassificationEvaluator()
          .setLabelCol("survived")
          .setRawPredictionCol("prediction")
          .setMetricName("areaUnderROC")
      )
    val model = crossValidator.fit(featureDF)
       // model.save("src/main/data/model/model1")
    //val Array(trainDF,testDF) = featureDF.randomSplit(Array(0.7,0.3))
    //val model = pipeline.fit(featureDF)
   // val prediction = model.transform(testDF)
   // prediction.show(20)
    //prediction.select("passengerId","survived","prediction").write.format("csv").save("src/main/data/result.csv")
   // prediction.write.format("csv").save("src/main/data/result.csv")

    //处理test数据集，测试数据是kaggle官网的测试数据，按照上面的逻辑再处理一遍
    val testDF:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("src/main/data/test.csv")
    testDF.printSchema()


   // val avgAge: Double = testDF.select($"Age").agg("Age" -> "avg").first().getDouble(0)

    val testRDD: RDD[FeatureDF] = testDF
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
      val Fare = if(row.get(8) != null) row.getDouble(8) else 0.0
      //      Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      //   Array(passengerId,survived,Pclass,sex,age,familyCate,Fare,cabin,embarked)
      FeatureDF(passengerId,0,titleName,Pclass,sex,age,familyCate,Fare,cabin,embarked)
    }
    )
    val testData = testRDD.toDF()

    val prediction: DataFrame = model.transform(testData)
    prediction.show(10)
    prediction.select("passengerId","prediction").write.format("csv").save("src/main/data/result4.csv")



    spark.stop()
  }

  //从姓名中提取特征
  def title_mapDict(name:String): String ={
     if(name.contains("Capt") || name.contains("Col")||name.contains("Major")||name.contains("Dr")||name.contains("Rev")){
      return "Officer"
    }
    if(name.contains("Jonkheer") || name.contains("Don")||name.contains("Sir")||name.contains("the Countess")||name.contains("Dona")||name.contains("Lady")){

      return "Royalty"
    }
    if(name.contains("Mme") || name.contains("Ms")||name.contains("Mrs")){
      return "Mrs"
    }
    if(name.contains("Mlle") || name.contains("Mlle")){
      return "Miss"
    }
    if(name.contains("Mr")){
      return "Mr"
    }
    if(name.contains("Master")){
      return "Master"
    }
    return "Anyone"
  }
}
case class FeatureDF(passengerId:Int ,survived: Int,titleName:String,Pclass:Int,sex:String,age:Double,familyCate:Int,Fare:Double,cabin:String,embarked:String)