from pyspark.sql.functions import col, concat_ws
from pyspark.sql import SparkSession
from operator import add
import sys
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import split
from pyspark.sql.types import ArrayType, StringType

if __name__ == "__main__":

    spark = SparkSession.builder.appName('FP-Growth approach').getOrCreate()
    sc=spark.sparkContext
    inputPath = 'hdfs://master:9000/user/user/'+sys.argv[2]
    outputPath = 'hdfs://master:9000/user/user/'+sys.argv[3]
    threshold = int(sys.argv[1])
    # data = spark.read.text("/content/data.txt").select(split("value", " ").alias("items"))
    # dataFile = sc.textFile("hdfs://master:9000/user/user/test.txt").select(split("value", " ").alias("items"))
    dataFile = spark.read.text(inputPath).select(split("value", " ").alias("items"))
    rows = dataFile.count()
    support = float(threshold/rows)
    #print('NAIII--------------------------',support,threshold,dataFile.count())
    # lines = spark.read.text('data.txt').select(split("value", ",").alias("items"))
    fpGrowth = FPGrowth(itemsCol="items", minSupport=support)
    itemsets = fpGrowth.fit(dataFile)
    freq_itemsets = itemsets.freqItemsets
    # freq_itemsets = itemsets.freqItemsets.filter(itemsets.freqItemsets.freq >= threshold)
    freq_itemsets = freq_itemsets.withColumn('items', concat_ws(',', 'items'))
    #freq_itemsets.rdd.saveAsTextFile(outputFileName)
    freq_itemsets.coalesce(1).write.format('csv').save(outputPath)
    # freq_itemsets.show()
    # freq_itemsets.withColumn('items', concat_ws(',', 'items')).write.csv(outputFileName)
    # Display frequent itemsets.
    # model.freqItemsets.show( )
    sc.stop()
