package mllib.examples.clustering
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.clustering.KMeans


object KMeansDF {
  
   def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KMeansDF").setMaster("local");
    val sc = new SparkContext(conf)
  
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    
    
    // Loads data.
    val dataset = sqlContext.read.format("libsvm").load("sample_kmeans_data.txt")
    
    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)
    
    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    
    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  } 
  
}