package mllib.examples.collaborativefiltering

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession

object CollaborativeFilteringDF {
   
  
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
    def parseRating(str: String): Rating = {
      val fields = str.split("::")
      assert(fields.size == 4)
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
    }
  
   def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("GaussianMixtureModelDF").setMaster("local");
    val sc = new SparkContext(conf)
  
    val spark = new org.apache.spark.sql.SQLContext(sc)
    
    /* val spark = SparkSession
      .builder
      .appName("ALSExample")
      .getOrCreate()*/
    import spark.implicits._
    
    
    
   
    val ratings = spark.read.textFile("sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
    
    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)
    
    // Evaluate the model by computing the RMSE on the test data
    val predictions = model.transform(test)
    
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
  }
}



