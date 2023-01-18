package com.paulok;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SparkUtils {

    public static SparkSession getSparkSession() {
        return SparkSession.builder()
                .appName("Diabetes Indicator")
                .master("local[*]")
                .getOrCreate();
    }

    public static Dataset<Row> readCsvFileWithHeaders(String filePath) {
        return getSparkSession()
                .read()
                .format("csv")
                .option("header", "true")
                .load(filePath)
                .cache();
    }
}
