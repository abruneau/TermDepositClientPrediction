from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import matplotlib.pyplot as plt
import seaborn as sb


conf = SparkConf().setAppName("term-deposit-prediction")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# # Data Ingestion
# We need to load data from a file in to a Spark DataFrame.
# Each row is an observed customer, and each column contains
# attributes of that customer.

schema = StructType([StructField("age", IntegerType(), True),     
  StructField("job", StringType(), True),     
  StructField("marital", StringType(), True),     
  StructField("education", StringType(), True),     
  StructField("default", StringType(), True),     
  StructField("balance", IntegerType(), True),     
  StructField("housing", StringType(), True),     
  StructField("loan", StringType(), True),     
  StructField("contact", StringType(), True),     
  StructField("day", IntegerType(), True),     
  StructField("month", StringType(), True),     
  StructField("duration", IntegerType(), True),     
  StructField("campaign", IntegerType(), True),
  StructField("pdays", IntegerType(), True),
  StructField("previous", LongType(), True),
  StructField("poutcome", StringType(), True),
  StructField("clientSubscribed", StringType(), True)
])

raw_bank_data = sqlContext.read.format('com.databricks.spark.csv').option("delimiter",";").option("header", "true").load('/tmp/bank-full.csv', schema = schema)
raw_bank_data.head()

count = raw_bank_data.count()
did_not_subscribed = raw_bank_data.filter(col("clientSubscribed") == 'no').count()
did_subscribe = raw_bank_data.filter(col("clientSubscribed") == 'yes').count()

print("total: %d, %d subscribed, %d did not" % (count, did_subscribe, did_not_subscribed))

# # Data Enrichment
# Data will be enriched the “isContactedRecently” flag based on pdays field in dataset. Condition would be:
# If pDays < 100 then isContactedRecently  = true else isContactedRecently  = false.

bank_data_enrich = raw_bank_data.withColumn("isContactedRecently", when((col("pdays") < 100) & (col("pdays") > 0), "true").otherwise("false"))


# # Data exploration
# The data vizualization workflow for large data sets is usually:
# 
# * Sample data so it fits in memory on a single machine.
# * Examine single variable distributions.
# * Examine joint distributions and correlations.
# * Look for other types of relationships.

sample_data = bank_data_enrich.sample(False, 0.5, 83).toPandas()

# We want to examine the distribution of our features, so start with them one at a time.

sb.distplot(sample_data['age'], kde=False)

# We can examine feature differences in the distribution of our features when we condition (split) our data in whether they subscribed or not.

sb.boxplot(x="clientSubscribed", y="age", data=sample_data)

# Looking at joint distributions of data can also tell us a lot, particularly about redundant features.

example_numeric_data = sample_data[["age", "balance", "duration", "pdays", "clientSubscribed"]]

sb.pairplot(example_numeric_data, hue="clientSubscribed")

# # Data Cleansing 
# Filter out the records with job as “unknown”.

clean_bank_data = bank_data_enrich.filter(col("job") != "unknown")

# # Logistic Regression
# We need to:
# * Code features that are not already numeric
# * Gather all features we need into a single column in the DataFrame.
# * Split labeled data into training and testing set
# * Fit the model to the training data.

numeric_col = ["age", "balance", "duration", "pdays"]
categorical_col =["job", "marital", "poutcome"]
categorical_col_indexed = map(lambda x: x+"_indexed", categorical_col)

label_indexer = StringIndexer(inputCol = 'clientSubscribed', outputCol = 'label').fit(clean_bank_data)
categorical_indexer = map(lambda x: StringIndexer(inputCol = x, outputCol = x+'_indexed').fit(clean_bank_data), categorical_col)

assembler = VectorAssembler(
    inputCols = categorical_col_indexed + numeric_col,
    outputCol = 'features')

# We can now define our classifier and pipeline. With this done, we can split our labeled data in train and test sets and fit a model.
# 
# To train the decision tree, give it the feature vector column and the label column.


lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.01)

stages = [label_indexer] + categorical_indexer + [assembler, lr]

pipeline = Pipeline(stages=stages)

(train, test) = clean_bank_data.randomSplit([0.7, 0.3])

model = pipeline.fit(train)

# The most important question to ask:
#     
#     Is my predictor better than random guessing?
# 
# How do we quantify that?

# Measure the area under the ROC curve, abreviated to AUROC.
# 
# Plots True Positive Rate vs False Positive Rate for binary classification system
# 
# [More Info](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
# 
# TL;DR for AUROC:
#     * .90-1 = excellent (A)
#     * .80-.90 = good (B)
#     * .70-.80 = fair (C)
#     * .60-.70 = poor (D)
#     * .50-.60 = fail (F)
# 

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
"The AUROC is %s" % (auroc)