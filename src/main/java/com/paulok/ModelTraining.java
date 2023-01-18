package com.paulok;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class SparkApplication {

    private static final String DATA_FILE_PATH = "diabetes.csv";
    private static final String FEATURES_DIVIDED_BY_COMMA = "HighBP,HighChol,CholCheck,BMI,Smoker,Stroke," +
            "HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth," +
            "MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income";
    private static final String[] FEATURES = FEATURES_DIVIDED_BY_COMMA.split(",");

    public static void main(String[] args) throws IOException {
        String filePath = DATA_FILE_PATH;
        if (args.length > 0) {
            filePath = args[0];
        }

        Logger.getLogger("org").setLevel(Level.ERROR);
        SparkSession sparkSession = SparkSession.builder()
                .appName("Diabetes Indicator")
                .master("local")
                .getOrCreate();

        Dataset<Row> ds = sparkSession.read().format("csv").option("header", "true").load(filePath).cache();
        ds = ds.na().drop();

        ds = ds.withColumn("Diabetes_binary", ds.col("Diabetes_binary").cast("double"));
        for (String feature: FEATURES) {
            ds = ds.withColumn(feature, ds.col(feature).cast("double"));
        }

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(FEATURES)
                .setOutputCol("features");

        Dataset<Row>[] splits = ds.randomSplit(new double[] {0.8, 0.1, 0.1});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> crossValidationData = splits[1];
        Dataset<Row> testData = splits[2];

        RandomForestClassifier rfc = new RandomForestClassifier()
                .setLabelCol("Diabetes_binary")
                .setFeaturesCol("features")
                .setMaxDepth(12)
                .setMaxBins(4096)
                .setNumTrees(20)
                .setFeatureSubsetStrategy("auto");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, rfc});
        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictionsForTrainingData = model.transform(trainingData);
        Dataset<Row> predictionsForCrossValidationData = model.transform(crossValidationData);
        Dataset<Row> predictionsForTestData = model.transform(testData);

        System.err.println("Training data 0 rate: " + (predictionsForTrainingData.filter("prediction == 0").count() / (double) trainingData.filter("Diabetes_binary == 0").count()));
        System.err.println("Training data 1 rate: " + (predictionsForTrainingData.filter("prediction == 1").count() / (double) trainingData.filter("Diabetes_binary == 1").count()));
        System.err.println("Cross-validation data 0 rate: " + (predictionsForCrossValidationData.filter("prediction == 0").count() / (double) crossValidationData.filter("Diabetes_binary == 0").count()));
        System.err.println("Cross-validation data 1 rate: " + (predictionsForCrossValidationData.filter("prediction == 1").count() / (double) crossValidationData.filter("Diabetes_binary == 1").count()));
        System.err.println("Test data 0 rate: " + (predictionsForTestData.filter("prediction == 0").count() / (double) testData.filter("Diabetes_binary == 0").count()));
        System.err.println("Test data 1 rate: " + (predictionsForTestData.filter("prediction == 1").count() / (double) testData.filter("Diabetes_binary == 1").count()));

        model.write().overwrite().save("target/model");
    }
}
