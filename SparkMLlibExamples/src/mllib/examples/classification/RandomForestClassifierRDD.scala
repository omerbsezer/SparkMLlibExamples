package mllib.examples.classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.{SparkConf, SparkContext}
import scala.io.Source // for url
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object RandomForestClassifierRDD  {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassifierRDD").setMaster("local");
    val sc = new SparkContext(conf)
  
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
      
    //Read the (CSV) file in as a sequence of lines.
    val csv = Source.fromURL("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data").mkString.split("\\r?\\n")
   
    //convert the data to an RDD
    val rdd = sc.parallelize(csv)
    
    /*
    Split each line (using “,”) of the CSV file into separate fields
    Some of the rows in the dataset contain missing values in the seventh field. Remove those.
    The first column contains an ID. Drop it.
    Convert all remaining values to floating point numbers*/

    val data = rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))
    
    //Create a set of LabeledPoint objects that contain a set of feature values and a label (the diagnosis).
    //The dataset uses 2 to represent a benign diagnosis and 4 to represent a malignant one. We’ll use 0 and 1, respectively
    val labeledPoints = data.map(x => LabeledPoint(if (x.last == 4) 1 else 0,  Vectors.dense(x.init)))
    
    
    val splits = labeledPoints.randomSplit(Array(0.7, 0.3), seed = 5043l)

    //split the data into training and test datasets—70% for training and 30% for testing.
    val trainingData = splits(0)
    val testData = splits(1)
    
    //Set up the model’s hyperparameters
    //We have a classification problem, and we’ll use 20 trees each having a maximum depth of three. The other parameters are pretty standard.
    val algorithm = Algo.Classification
    val impurity = Gini
    val maximumDepth = 3
    val treeCount = 20
    val featureSubsetStrategy = "auto"
    val seed = 5043
    
    val model = RandomForest.trainClassifier(trainingData, new Strategy(algorithm, impurity, maximumDepth), treeCount, featureSubsetStrategy, seed)
    
    //Using the test dataset we’ll generate a number of predictions and cross reference them with the actual diagnoses.
    val labeledPredictions = testData.map { labeledPoint =>
        val predictions = model.predict(labeledPoint.features)
        (labeledPoint.label, predictions)
    }
    
    //MLlib makes it easy to evaluate the effectiveness of our classifier.
    val evaluationMetrics = new MulticlassMetrics(labeledPredictions.map(x => (x._1, x._2)))
   
    println("confusionMatrix:\n" + evaluationMetrics.confusionMatrix)
   
    println("precision:" +evaluationMetrics.precision)
   
    
  }
}
