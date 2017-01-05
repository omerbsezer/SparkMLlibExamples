package mllib.examples.regression

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object LinearRegressionDF {
  def main(args: Array[String]) = {
     //Start the Spark context
    val conf = new SparkConf().setAppName("LinearRegressionDF").setMaster("local")
    val sc = new SparkContext(conf);
    val sqlContext = new SQLContext(sc)

    val training:DataFrame = sqlContext.read.format("libsvm").load("sample_linear_regression_data.txt")
    // Load training data
    //val training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")
    
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    
    // Fit the model
    val lrModel = lr.fit(training)
    
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    
    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}