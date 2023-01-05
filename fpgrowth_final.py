from pyspark.sql.functions import concat_ws
from pyspark.sql import SparkSession
import sys
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import split

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('Usage: fpgrowth <threshold> <inputFromDfs> <outputFromDfs>', file=sys.stderr)
        exit(-1)
    spark = SparkSession.builder.appName('FP-Growth approach').getOrCreate()
    sc = spark.sparkContext

    # Declare input and output from DFS
    inputPath = 'hdfs://master:9000/user/bigdata/' + sys.argv[2]
    outputPath = 'hdfs://master:9000/user/bigdata/' + sys.argv[3]

    threshold = int(sys.argv[1])
    # Save data to a DataFrame
    dataFile = spark.read.text(inputPath).select(split('value', ' ').alias('items'))
    # Count total rows (transactions) that are in the DataFrame
    rows = dataFile.count()
    # Convert threshold to support
    support = float(threshold / rows)
    # Find frequent itemsets
    fpGrowth = FPGrowth(itemsCol='items', minSupport=support)
    itemsets = fpGrowth.fit(dataFile)
    freq_itemsets = itemsets.freqItemsets
    # Transform DataFrame so it can be written to csv
    freq_itemsets = freq_itemsets.withColumn('items', concat_ws(',', 'items'))
    # Combine all result csv parts to one and save it to the output path
    freq_itemsets.coalesce(1).write.format('csv').save(outputPath)
    sc.stop()
