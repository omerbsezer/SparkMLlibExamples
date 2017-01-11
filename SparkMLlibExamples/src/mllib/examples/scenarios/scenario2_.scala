package mllib.examples.scenarios

import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object scenario2_ {
 
 def main(args: Array[String]) {
     //Start the Spark context
    val conf = new SparkConf().setAppName("scenario2-regression").setMaster("local")
    val sc = new SparkContext(conf);
    val sqlContext = new SQLContext(sc)
    // Load and parse the data
    val data = sc.textFile("sample_data_scenario2.txt")
    val parsedData = data.map { line =>
      val parts = line.split(';')
      LabeledPoint(parts.last.toDouble, Vectors.dense(parts.take(11).map(_.toDouble)))
    }
    
    // Build linear regression model
    var regression = new LinearRegressionWithSGD().setIntercept(true)
    regression.optimizer.setStepSize(0.001)
    val model = regression.run(parsedData)
    
    // Export linear regression model to PMML
    model.toPMML("exported_models/scenario2.xml")
    
    // Test model on training data
    // (quality: 5)
    var predictedValue = model.predict(Vectors.dense(7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4))
    println("predictedValue1:"+predictedValue ) 
    //  (quality: 7)
    predictedValue = model.predict(Vectors.dense(11.5,0.54,0.71,4.4,0.124,6,15,0.9984,3.01,0.83,11.8))
    println("predictedValue2:"+predictedValue ) 
    
    //source:https://github.com/selvinsource/spark-pmml-exporter-validator
   }
}