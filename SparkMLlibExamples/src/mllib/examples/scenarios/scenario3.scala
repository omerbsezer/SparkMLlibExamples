package mllib.examples.scenarios

import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors

import com.cloudera.sparkts.models.ARIMA


object scenario3 {
  
   case class SensorTelemetry(sensorId: String, timestamp: String, value: Double)
    def parseData(str: String): SensorTelemetry = {
      val fields = str.split(";")
      assert(fields.size == 3)
      SensorTelemetry(fields(0).toString, fields(1).toString, fields(2).toDouble)
    }
  
  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf().setAppName("scenario3").setMaster("local")
    val sc = new SparkContext(conf);
    val sqlContext = new SQLContext(sc)
    
   
    val spark = new org.apache.spark.sql.SQLContext(sc)
    
    import spark.implicits._
    
    val sensorTlmDF = spark.read.textFile("sensor_telemetries_data.txt").map(parseData).toDF()
    //sensorTlmDF.select('*).show()
  

   
   // val lines=sensorTlmDF.select("value").rdd
    

    //val lines = scala.io.Source.fromFile("sensor_telemetries_data2.txt").getLines()
    val lines = scala.io.Source.fromFile("arima_time_series_data.csv").getLines()
    val ts = Vectors.dense(lines.map(_.toDouble).toArray)
    val arimaModel = ARIMA.fitModel(1, 0, 1, ts)
    println("coefficients: " + arimaModel.coefficients.mkString(","))
    val forecast = arimaModel.forecast(ts, 20)
    println("forecast of next 20 observations: " + forecast.toArray.mkString(","))
    
  }
}