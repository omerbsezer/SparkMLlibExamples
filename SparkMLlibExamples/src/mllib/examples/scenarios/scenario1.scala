package mllib.examples.scenarios

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._


object scenario1 {
  
   def main(args: Array[String]) {
     //Start the Spark context
    val conf = new SparkConf().setAppName("scenario1").setMaster("local")
    val sc = new SparkContext(conf);
    val sqlContext = new SQLContext(sc)
    
   
    val spark = new org.apache.spark.sql.SQLContext(sc)
    
    import spark.implicits._
   
     val schema = Seq("sensorId", "timestamp", "value")
     val data = Seq(
            ("sensor1", "2017-01-02-10:00:00", 10),
            ("sensor1", "2017-01-02-10:01:00", 50),
            ("sensor1", "2017-01-02-10:02:00", 20),
            ("sensor1", "2017-01-02-10:03:00", 30),
            ("sensor1", "2017-01-02-10:04:00", 55),
            ("sensor1", "2017-01-02-10:05:00", 24),
            ("sensor1", "2017-01-02-10:06:00", 37),
            ("sensor1", "2017-01-02-10:07:00", 40),
            ("sensor1", "2017-01-02-10:08:00", 70),
            ("sensor1", "2017-01-02-10:09:00", 40),
            ("sensor1", "2017-01-02-10:10:00", 10),
            ("sensor1", "2017-01-02-10:11:00", 50),
            ("sensor1", "2017-01-02-10:12:00", 70),
            ("sensor1", "2017-01-02-10:13:00", 90)
          )
    
    val dft = sc.parallelize(data).toDF(schema: _*)
    
    dft.select('*).show
    
    // Calculate the moving average with 5 values before 2 and after 2
    val dataWithWindow = Window.partitionBy("sensorId").rowsBetween(-2, 2)
    
    //Calculate average
    val allData = Window.partitionBy("sensorId")
    
    // Calculate cumulative sum
    val dataForCS = Window.partitionBy("sensorId").orderBy("value").rowsBetween(Long.MinValue, 0)
    
    
    //showTable
    val movingAverageTable = dft.select($"sensorId",$"timestamp",$"value",avg($"value").over(dataWithWindow),avg($"value").over(allData),sum($"value").over(dataForCS))
    movingAverageTable.show
    
    
    
    
   
  
   
  }
}