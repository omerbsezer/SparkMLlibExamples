package mllib.examples.clustering

import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object GaussianMixtureModelDF {
    
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GaussianMixtureModelDF").setMaster("local");
    val sc = new SparkContext(conf)
  
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    
    
    // Loads data.
    val dataset = sqlContext.read.format("libsvm").load("sample_kmeans_data.txt")

    
    // Trains Gaussian Mixture Model
    val gmm = new GaussianMixture()
      .setK(2)
    val model = gmm.fit(dataset)
    
    // output parameters of mixture model model
    for (i <- 0 until model.getK) {
      println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
          s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
    }
  }
}



