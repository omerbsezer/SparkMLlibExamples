package mllib.examples.classification

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object NaiveBayesClassifierDF {
  def main(args: Array[String]) = {
     //Start the Spark context
    val conf = new SparkConf().setAppName("NaiveBayesClassifierDF").setMaster("local")
    val sc = new SparkContext(conf);
    val sqlContext = new SQLContext(sc)

    //load data
    val data:DataFrame = sqlContext.read.format("libsvm").load("sample_libsvm_data.txt")
    // Load the data stored in LIBSVM format as a DataFrame.
   
    
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    
    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainingData)
    
    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()
    
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)
  }
}



