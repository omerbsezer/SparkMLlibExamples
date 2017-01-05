package mllib.examples.clustering

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object LatentDirichletAllocationDF {
    
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LatentDirichletAllocationDF").setMaster("local");
    val sc = new SparkContext(conf)
  
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    
    
    // Loads data.
    val dataset = sqlContext.read.format("libsvm").load("sample_lda_libsvm_data.txt")
  
    
    // Trains a LDA model.
    val lda = new LDA().setK(10).setMaxIter(10)
    val model = lda.fit(dataset)
    
    val ll = model.logLikelihood(dataset)
    val lp = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound bound on perplexity: $lp")
    
    // Describe topics.
    val topics = model.describeTopics(3)
    println("The topics described by their top-weighted terms:")
    topics.show(false)
    
    // Shows the result.
    val transformed = model.transform(dataset)
    transformed.show(false)
  }
}



