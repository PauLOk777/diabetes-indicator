package com.paulok;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class SparkApplication {

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        SparkConf conf = new SparkConf().setMaster("local").setAppName("Spark test");
        JavaSparkContext sc = new JavaSparkContext(conf);
        System.out.println("1");
    }
}
