from pyspark.sql.functions import col, concat_ws
from pyspark.sql import SparkSession
from operator import add
import sys
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import split

if __name__ == "__main__":

    spark = SparkSession.builder.appName('FP-Growth approach').getOrCreate()
    sc=spark.sparkContext
    threshold = int(sys.argv[1])
    # data = spark.read.text("/content/data.txt").select(split("value", " ").alias("items"))
    # dataFile = sc.textFile("hdfs://master:9000/user/user/test.txt").select(split("value", " ").alias("items"))
    dataFile = spark.read.text("hdfs://master:9000/user/user/test").select(split("value", " ").alias("items"))

    # lines = spark.read.text('data.txt').select(split("value", ",").alias("items"))
    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.0)
    itemsets = fpGrowth.fit(dataFile)
    freq_itemsets = itemsets.freqItemsets.filter(itemsets.freqItemsets.freq >= threshold)
    freq_itemsets.show()
    freq_itemsets.withColumn('items', concat_ws(',', 'items')).write.csv('/home/bigdata/kalhspera')
    # Display frequent itemsets.
    # model.freqItemsets.show()
