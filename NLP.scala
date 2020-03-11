// Databricks notebook source
// DBTITLE 1,Natural Language Processing and Wine Review Analysis
import org.apache.spark._
import org.apache.spark.sql.expressions.Window

//import libs 
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature._
import spark.sqlContext.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC !pip install nltk
// MAGIC import nltk
// MAGIC nltk.download("all")
// MAGIC from nltk.stem import SnowballStemmer
// MAGIC import pyspark.sql.functions as F
// MAGIC from pyspark.sql import types
// MAGIC from pyspark.sql import Row

// COMMAND ----------

// DBTITLE 1,Select Most Relevant and Top Classes
val DF = spark.sql("select * from winemag")

val window = Window.partitionBy("variety")

val WineDF = DF.withColumn("frequency", count("variety").over(window))
        .orderBy(desc("frequency"))
        .where(col("frequency")>3000) //Threshold set at 2000(need to justify)

//WineDF.count()

// COMMAND ----------

//Total classes 
val wine_class = WineDF.select("variety").distinct()
wine_class.show
wine_class.count()

// COMMAND ----------

// DBTITLE 1,Drop Nulls and Columns
//Remove unecessary columns and strip records with null values

//val concatDF = WineDF.withColumn("description",concat(WineDF("description"),lit(" "),WineDF("taster_name")))
//                                        ,lit(" "),WineDF("country")))

val dropcolDF = WineDF.drop("title").drop("region_1").drop("region_2")
               .drop("taster_name")
               .drop("taster_twitter_handle")      
               .drop("winery").drop("designation").drop("frequency").drop("_c0")
               .na.drop()   //Strip null values
               .withColumn("id",monotonically_increasing_id()) //re-index the data

//Strip Special Characters
val cleanedDF = dropcolDF.select(dropcolDF.columns.map(c => regexp_replace(dropcolDF(c), """[^A-Za-z0-9\s]+""", "").alias(c)): _*)

//cleanedDF.count()
//cleanedDF.show()

// COMMAND ----------

// display(cleanedDF.select("concatenate"))

// COMMAND ----------

// DBTITLE 1,Set Tokenizer and Strip Stopwords
//Tokenize
val Tokenize = new Tokenizer()
                .setInputCol("description")
                .setOutputCol("description_token")

//Remove stop words
val Stopwordsremover = new StopWordsRemover()
                .setInputCol(Tokenize.getOutputCol)
                .setOutputCol("filtered")

val pipeline1 = new Pipeline()
              .setStages(Array(Tokenize, Stopwordsremover))

val Stopwords = pipeline1.fit(cleanedDF).transform(cleanedDF)

//Create Tempview
Stopwords.createTempView("Stopwords")

//Stopwords.show()
//Stopwords.count()

// COMMAND ----------

// DBTITLE 1,Engage Stemmer
// MAGIC %python
// MAGIC from pyspark.sql.types import *
// MAGIC 
// MAGIC #Call table to python
// MAGIC Stopwords = spark.table("Stopwords")
// MAGIC 
// MAGIC #Use snowball
// MAGIC stemmer = SnowballStemmer('english')
// MAGIC 
// MAGIC #UDF to stem each token
// MAGIC udf1 = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
// MAGIC Stemmed = Stopwords.withColumn("stemmed", udf1("filtered"))
// MAGIC 
// MAGIC #Cast datatypes
// MAGIC Stemmed = Stemmed.withColumn("points", Stemmed["points"].cast("int")).withColumn("price", Stemmed["price"].cast("int"))
// MAGIC 
// MAGIC Stemmed.createTempView("Stemmed")

// COMMAND ----------

// DBTITLE 1,Feature Extraction
//Call table in scala
val Stemmed = spark.table("Stemmed")

//Indexing Country (Categorical Feature)
val countryIndex = new StringIndexer()
              .setInputCol("country")
              .setOutputCol("countryIndex")

// val tasterhandle = new StringIndexer()
//               .setInputCol("taster_twitter_handle")
//               .setOutputCol("tasterhandle")

//Feature Extractors
//CountVectorizer Model - - (TF followed by IDF)
val countVec = new CountVectorizer()
                .setInputCol("stemmed")
                .setOutputCol("countvec")
              //  .setMinDF(5)
              //  .setMinTF(5)
                  .setVocabSize(4000)

//HashTF Model - (TF followed by IDF) 
val hashTF = new HashingTF()
             .setInputCol("stemmed")
             .setOutputCol("hashtf")
             .setNumFeatures(4000)

val IDF = new IDF()
           .setInputCol(countVec.getOutputCol)
           .setOutputCol("tfidf")

//Word2Vec Model - (Uses word similarity)

// val word2Vec = new Word2Vec()
//                 .setInputCol("stemmed")
//                 .setOutputCol("word2vec")

//Choose which feature extractor to use and add it to pipeline
val pipeline2 = new Pipeline()
              .setStages(Array(countryIndex, countVec, IDF))

val FeatureExtractor = pipeline2.fit(Stemmed).transform(Stemmed)

//FeatureExtractor.show()

// COMMAND ----------

// DBTITLE 1,Vector Assemble
//Vector Assembler
//Choose the require features and iterate multiple times

val assembler =  new VectorAssembler()
                .setInputCols(Array("tfidf","countryIndex","points"))  //or use word2vec
                .setOutputCol("features")

val AssembledDF = assembler.transform(FeatureExtractor).drop("description")

// COMMAND ----------

// DBTITLE 1,Dimensionality Reduction
//Dimensionality Reduction

val pcafeatures = new PCA()
                  .setInputCol("features")
                  .setOutputCol("pcafeatures")
                  .setK(500)  //Choose appropriate number of features

val PCADF = pcafeatures.fit(AssembledDF).transform(AssembledDF)

// COMMAND ----------

// DBTITLE 1,Prepare Label, Features and Test/Train Split
//Test-Train Split

val FeatureDF = PCADF.select("pcafeatures","variety")
val Array(train_data,test_data) = FeatureDF.randomSplit(Array(0.7, 0.3), seed = 12345)

//FeatureDF.show()

// Prep Index labels and features
val labelIndexer = new StringIndexer()
                     .setInputCol("variety")
                     .setOutputCol("varietyIndex")
                     .fit(FeatureDF)

val featureIndexer = new VectorIndexer()
                      .setInputCol("pcafeatures")
                      .setOutputCol("featureIndex")
                      .fit(FeatureDF)

//Scale Feature for Naive Bayes
val featureIndexer1 = new MinMaxScaler()
                  .setInputCol("pcafeatures")
                  .setOutputCol("featureIndex1")
                   .setMax(1)
                   .setMin(0)

val labelConverter = new IndexToString()
                      .setInputCol("prediction")
                      .setOutputCol("predictedLabel")
                      .setLabels(labelIndexer.labels)

// COMMAND ----------

// DBTITLE 1,Model 1 - Logistic Regression
//Logistic Regression Model

val LR = new LogisticRegression()
         .setFeaturesCol("featureIndex")   //setting features column
         .setLabelCol("varietyIndex")

val LR_pipeline = new Pipeline()
                .setStages(Array(labelIndexer, featureIndexer, LR, labelConverter))

val LR_model = LR_pipeline.fit(train_data)

val LR_predictions = LR_model.transform(test_data)

// COMMAND ----------

// DBTITLE 1,Logistic Regression Evaluation
val LR_evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("varietyIndex")
                  .setPredictionCol("prediction")
                  .setMetricName("accuracy")

val LR_testaccuracy = LR_evaluator.evaluate(LR_predictions)

println("Test Error for Log Regression = " + (1.0 - LR_testaccuracy))

// COMMAND ----------

// DBTITLE 1,Logistic Regression - Metrics
val LR_predictions1 = LR_predictions.select("prediction", "varietyIndex")

val LR_RDD = LR_predictions1.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val LR_metrics= new MulticlassMetrics(LR_RDD)

println(s"Weighted precision: ${LR_metrics.weightedPrecision}")
println(s"Weighted recall: ${LR_metrics.weightedRecall}")
println(s"Weighted F1 score: ${LR_metrics.weightedFMeasure}")
println(s"Accuracy: ${LR_metrics.accuracy}")

// COMMAND ----------

// DBTITLE 1,Logistic Regression - Check for Overfit
val LR_train = LR_model.transform(train_data)

val LR_trainaccuracy = LR_evaluator.evaluate(LR_train)

println("Train Error for Log Regression = " + (1.0 - LR_trainaccuracy))

// COMMAND ----------

// DBTITLE 1,Logistic Regression - HyperParameter Tuning and Cross Validation
//Logistic Regression HyperParameter Tuning and Cross Validation

val LR_paramGrid = new ParamGridBuilder()
                .addGrid(LR.regParam, Array(0,0.1))
                .addGrid(LR.elasticNetParam, Array(0,0.1,0.3))
                .addGrid(LR.maxIter, Array(70,100))
                .build()

val LR_CrossValidation = new CrossValidator()
           .setEstimator(LR_pipeline)
           .setEvaluator(LR_evaluator)
           .setEstimatorParamMaps(LR_paramGrid)
           .setNumFolds(3)

// COMMAND ----------

val LR_CVmodel = LR_CrossValidation.fit(train_data)

// COMMAND ----------

val LR_CVpredictions = LR_CVmodel.transform(test_data)

// COMMAND ----------

val LR_CVaccuracy = LR_evaluator.evaluate(LR_CVpredictions)

println("Cross Validated Test Error for Logistic Regression = " + (1.0 -  LR_CVaccuracy))

// COMMAND ----------

// DBTITLE 1,Logistic Regression CV - Check for Overfit
val LR_CVtrain = LR_model.transform(train_data)

val LR_CVtrainaccuracy = LR_evaluator.evaluate(LR_CVtrain)

println("Cross Validated Train Error for Logistic Regression = " + (1.0 -  LR_CVtrainaccuracy))

// COMMAND ----------

// DBTITLE 1,Logistic Regression CV - Metrics
val LR_CVpredictions1 = LR_CVpredictions.select("prediction", "varietyIndex")

val LR_CVRDD = LR_CVpredictions1.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val LR_CVmetrics= new MulticlassMetrics(LR_CVRDD)

println(s"Weighted precision: ${LR_CVmetrics.weightedPrecision}")
println(s"Weighted recall: ${LR_CVmetrics.weightedRecall}")
println(s"Weighted F1 score: ${LR_CVmetrics.weightedFMeasure}")
println(s"Accuracy: ${LR_CVmetrics.accuracy}")

// COMMAND ----------

// DBTITLE 1,Logistic Regression ROC
// Curve Plotting

val LR1 = LR_predictions.select(col("featureIndex"),col("prediction"),col("predictedLabel"),col("variety"),col("varietyIndex"),col("probability").as("prob"))

import org.apache.spark.ml.linalg.DenseVector

//labelConverter.getLabels

val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
val toArrUdf = udf(toArr)
val Table1 = LR1.withColumn("probability",toArrUdf('prob))
Table1.createTempView("Table")

// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC import numpy as np
// MAGIC import matplotlib.pyplot as plt
// MAGIC from itertools import cycle 
// MAGIC from sklearn import svm, datasets 
// MAGIC from sklearn.metrics import roc_curve, auc
// MAGIC from sklearn.model_selection import train_test_split
// MAGIC from sklearn.preprocessing import label_binarize 
// MAGIC from sklearn.multiclass import OneVsRestClassifier 
// MAGIC from scipy import interp 
// MAGIC 
// MAGIC Table = spark.table("Table")
// MAGIC ovr_prob = Table.toPandas()
// MAGIC 
// MAGIC labels=ovr_prob["varietyIndex"].to_frame()
// MAGIC one_hot = pd.get_dummies(labels['varietyIndex'])
// MAGIC 
// MAGIC # Drop column B as it is now encoded
// MAGIC labels=labels.drop('varietyIndex',axis=1)
// MAGIC 
// MAGIC # Join the encoded df
// MAGIC labels=labels.join(one_hot)
// MAGIC y_score=ovr_prob["probability"].values
// MAGIC y_test=labels.values
// MAGIC y_score2=np.array([np.array(i) for i in y_score])
// MAGIC 
// MAGIC 
// MAGIC fpr=dict()
// MAGIC 
// MAGIC tpr=dict() 
// MAGIC 
// MAGIC roc_auc=dict()
// MAGIC 
// MAGIC for i in range(10):
// MAGIC   fpr[i], tpr[i], _ =roc_curve(y_test[:, i], y_score2[:, i]) 
// MAGIC   roc_auc[i] =auc(fpr[i], tpr[i])
// MAGIC 
// MAGIC # Compute micro-average ROC curve and ROC area 
// MAGIC fpr["micro"], tpr["micro"], _=roc_curve(y_test.ravel(), y_score2.ravel()) 
// MAGIC roc_auc["micro"] =auc(fpr["micro"], tpr["micro"]) 
// MAGIC 
// MAGIC all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
// MAGIC 
// MAGIC # Then interpolate all ROC curves at this points
// MAGIC mean_tpr=np.zeros_like(all_fpr)
// MAGIC for i in range(10):
// MAGIC   mean_tpr+=interp(all_fpr, fpr[i], tpr[i])
// MAGIC 
// MAGIC  # Finally average it and compute AUC
// MAGIC mean_tpr/=10
// MAGIC fpr["macro"] =all_fpr
// MAGIC tpr["macro"] =mean_tpr
// MAGIC roc_auc["macro"] =auc(fpr["macro"], tpr["macro"])
// MAGIC 
// MAGIC # Plot all ROC curves
// MAGIC abc=plt.figure(figsize=(12,10))
// MAGIC lw=2
// MAGIC plt.plot(fpr["micro"], tpr["micro"],
// MAGIC          label='micro-average ROC curve (area = {0:0.2f})'
// MAGIC          .format(roc_auc["micro"]),
// MAGIC          color='deeppink', linestyle=':', linewidth=4)
// MAGIC 
// MAGIC plt.plot(fpr["macro"], tpr["macro"],
// MAGIC          label='macro-average ROC curve (area = {0:0.2f})'
// MAGIC          .format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
// MAGIC 
// MAGIC 
// MAGIC colors=cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
// MAGIC 
// MAGIC labels=['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend', 'Bordeaux-style Red Blend', 'Riesling', 'Sauvignon Blanc', 'Syrah', 'Rosé', 'Merlot']
// MAGIC 
// MAGIC for i,j, color in zip(range(10), labels, colors):
// MAGIC   plt.plot(fpr[i], tpr[i], color=color, lw=lw,
// MAGIC            label='ROC curve of class {0} (area = {1:0.2f})'
// MAGIC            .format(j, roc_auc[i]))
// MAGIC   
// MAGIC   
// MAGIC plt.plot([0, 1], [0, 1], 'k--', lw=lw)
// MAGIC plt.xlim([0.0, 1.0])
// MAGIC plt.ylim([0.0, 1.05])
// MAGIC plt.xlabel('False Positive Rate')
// MAGIC plt.ylabel('True Positive Rate')
// MAGIC plt.title('ROC Curve for the select Classes')
// MAGIC plt.legend(loc="lower right")
// MAGIC display(abc)

// COMMAND ----------

// DBTITLE 1,Model 2 - Decision Tree Classifier
//Decision Tree Model

val DT = new DecisionTreeClassifier()
            .setLabelCol("varietyIndex")
            .setFeaturesCol("featureIndex")

val DT_pipeline = new Pipeline()
                  .setStages(Array(labelIndexer, featureIndexer, DT, labelConverter))

val DT_model = DT_pipeline.fit(train_data)

// Make predictions
val DT_predictions = DT_model.transform(test_data)

// COMMAND ----------

// DBTITLE 1,Decision Tree Evaluation
val DT_evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("varietyIndex")
                  .setPredictionCol("prediction")
                  .setMetricName("accuracy")

val DT_testaccuracy = DT_evaluator.evaluate(DT_predictions)

println("Test Error for Decision Tree Classifier " + (1.0 - DT_testaccuracy))

// COMMAND ----------

// DBTITLE 1,Decision Tree - Check for Overfit
val DT_train = DT_model.transform(train_data)

val DT_trainaccuracy = DT_evaluator.evaluate(DT_train)

println("Train Error for Decision Tree Classifier = " + (1.0 - DT_trainaccuracy))

// COMMAND ----------

// DBTITLE 1,Decision Trees - Metrics
val DT_predict = DT_predictions.select("prediction", "varietyIndex")

val DT_RDD = DT_predict.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val DT_metrics= new MulticlassMetrics(DT_RDD)

println(s"Weighted precision: ${DT_metrics.weightedPrecision}")
println(s"Weighted recall: ${DT_metrics.weightedRecall}")
println(s"Weighted F1 score: ${DT_metrics.weightedFMeasure}")
println(s"Accuracy: ${DT_metrics.accuracy}")

// COMMAND ----------

// DBTITLE 1,Decision Tree - HyperParameter Tuning and Cross Validation
//Decision Tree HyperParameter Tuning and Cross Validation

val DT_paramGrid = new ParamGridBuilder()
                .addGrid(DT.maxDepth, Array(5,7))
                .addGrid(DT.impurity, Array("entropy","gini"))
                .addGrid(DT.maxBins, Array(25,35))
                .build()

val DT_CrossValidation = new CrossValidator()
           .setEstimator(DT_pipeline)
           .setEvaluator(DT_evaluator)
           .setEstimatorParamMaps(DT_paramGrid)
           .setNumFolds(3)

// COMMAND ----------

val DT_CVmodel = DT_CrossValidation.fit(train_data)

// COMMAND ----------

val DT_CVpredictions = DT_CVmodel.transform(test_data)

// COMMAND ----------

val DT_CVaccuracy = DT_evaluator.evaluate(DT_CVprediction)

println("Cross Validated Test Error for Decision Tree Classifier = " + (1.0 -  DT_CVaccuracy))

// COMMAND ----------

// DBTITLE 1,Decision Tree CV - Check for Overfit
val DT_CVtrain = DT_model.transform(train_data)

val DT_CVtrainaccuracy = DT_evaluator.evaluate(DT_CVtrain)

println("Cross Validated Train Error for Decision Tree Classifier = " + (1.0 -  DT_CVtrainaccuracy))

// COMMAND ----------

// DBTITLE 1,Decision Trees CV - Metrics
val DT_CVpredictions1 = DT_CVpredictions.select("prediction", "varietyIndex")

val DT_CVRDD = DT_CVpredictions1.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val DT_CVmetrics= new MulticlassMetrics(DT_CVRDD)

println(s"Weighted precision: ${DT_CVmetrics.weightedPrecision}")
println(s"Weighted recall: ${DT_CVmetrics.weightedRecall}")
println(s"Weighted F1 score: ${DT_CVmetrics.weightedFMeasure}")
println(s"Accuracy: ${DT_CVmetrics.accuracy}")

// COMMAND ----------

// DBTITLE 1,Model 3 - Random Forest Classifier
//RandomForest

val RF = new RandomForestClassifier()
        .setLabelCol("varietyIndex")
        .setFeaturesCol("featureIndex")
        .setNumTrees(20)

val RF_pipeline = new Pipeline()
                  .setStages(Array(labelIndexer, featureIndexer, RF, labelConverter))

val RF_model = RF_pipeline.fit(train_data)

// Make predictions
val RF_predictions = RF_model.transform(test_data)

// COMMAND ----------

// DBTITLE 1,Random Forest - Evaluation
val RF_evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("varietyIndex")
                  .setPredictionCol("prediction")
                  .setMetricName("accuracy")

val RF_testaccuracy = RF_evaluator.evaluate(RF_predictions)

println("Test Error for Random Forest Classifier " + (1.0 - RF_testaccuracy))

// COMMAND ----------

//RF_predictions.select("variety","varietyIndex","probability","prediction","predictedLabel").show()

// COMMAND ----------

// DBTITLE 1,Random Forest - Check for Overfit
val RF_train = RF_model.transform(train_data)

val RF_trainaccuracy = RF_evaluator.evaluate(RF_train)

println("Train Error for Random Forest Classifier = " + (1.0 - RF_trainaccuracy))

// COMMAND ----------

RF_predictions.select("variety","varietyIndex","probability","prediction","predictedLabel").show()

// COMMAND ----------

// DBTITLE 1,Random Forest - Metrics
val RF_predict = RF_predictions.select("prediction", "varietyIndex")

val RF_RDD = RF_predict.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val RF_metrics= new MulticlassMetrics(RF_RDD)

println(s"Weighted precision: ${RF_metrics.weightedPrecision}")
println(s"Weighted recall: ${RF_metrics.weightedRecall}")
println(s"Weighted F1 score: ${RF_metrics.weightedFMeasure}")
println(s"Accuracy: ${RF_metrics.accuracy}")

// COMMAND ----------

// DBTITLE 1,Random Forest - HyperParameter Tuning and Cross Validation
//Random Forest - HyperParameter Tuning and Cross Validation

val RF_ParamGrid = new ParamGridBuilder()
  .addGrid(RF.maxBins, Array(25,35))
  .addGrid(RF.maxDepth, Array(5,7))
  .addGrid(RF.impurity, Array("entropy", "gini"))
  .build()

// define cross validation stage to search through the parameters
// K-Fold cross validation with ClassificationEvaluator
val RF_CrossValidation = new CrossValidator()
  .setEstimator(RF_pipeline)
  .setEvaluator(RF_evaluator)
  .setEstimatorParamMaps(RF_ParamGrid)
  .setNumFolds(3)


// COMMAND ----------

val RF_CVmodel = RF_CrossValidation.fit(train_data)

// COMMAND ----------

val RF_CVpredictions = RF_CVmodel.transform(test_data)

// COMMAND ----------

val RF_CVaccuracy = RF_evaluator.evaluate(RF_CVpredictions)

println("Cross Validated Test Error for Random Forest Classifier = " + (1.0 - RF_CVaccuracy))

// COMMAND ----------

// DBTITLE 1,Random Forest CV - Check for Overfit
val RF_CVtrainprediction = RF_CVmodel.transform(train_data)

val RF_CVtrainaccuracy = RF_evaluator.evaluate(RF_CVtrainprediction)

println("Cross Validated Train Error for Random Forest Classifier = " + (1.0 - RF_CVtrainaccuracy))

// COMMAND ----------

// DBTITLE 1,Random Forest CV - Metrics
val RF_CVpredictions1 = RF_CVpredictions.select("prediction", "varietyIndex")

val RF_CVRDD = RF_CVpredictions1.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val RF_CVmetrics= new MulticlassMetrics(RF_CVRDD)

println(s"Weighted precision: ${RF_CVmetrics.weightedPrecision}")
println(s"Weighted recall: ${RF_CVmetrics.weightedRecall}")
println(s"Weighted F1 score: ${RF_CVmetrics.weightedFMeasure}")
println(s"Accuracy: ${RF_CVmetrics.accuracy}")

// COMMAND ----------

// DBTITLE 1,Model 4 - Naive Bayes Classifier
// Naive Bayes Classifier 

val NB = new NaiveBayes()
         .setFeaturesCol("featureIndex1")   //setting features column
         .setLabelCol("varietyIndex")

val NB_pipeline = new Pipeline()
                .setStages(Array(labelIndexer, featureIndexer1, NB, labelConverter))

val NB_model = NB_pipeline.fit(train_data)

val NB_predictions = NB_model.transform(test_data)

// COMMAND ----------

// DBTITLE 1,Naive Bayes - Evaluation
val NB_evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("varietyIndex")
                  .setPredictionCol("prediction")
                  .setMetricName("accuracy")

val NB_testaccuracy = NB_evaluator.evaluate(NB_predictions)

println("Test Error for Naive Bayes Classifier " + (1.0 - NB_testaccuracy))

// COMMAND ----------

// DBTITLE 1,Naive Bayes - Check for Overfit
val NB_train = NB_model.transform(train_data)

val NB_trainaccuracy = NB_evaluator.evaluate(NB_train)

println("Train Error for Naive Bayes Classifier = " + (1.0 - NB_trainaccuracy))

// COMMAND ----------

// DBTITLE 1,Naive Bayes - Metrics
val NB_predict = NB_predictions.select("prediction", "varietyIndex")

val NB_RDD = RF_predict.rdd.map{x=>(x.getAs[Double](0), x.getAs[Double](1))}

val NB_metrics= new MulticlassMetrics(NB_RDD)

println(s"Weighted precision: ${NB_metrics.weightedPrecision}")
println(s"Weighted recall: ${NB_metrics.weightedRecall}")
println(s"Weighted F1 score: ${NB_metrics.weightedFMeasure}")
println(s"Accuracy: ${NB_metrics.accuracy}")
