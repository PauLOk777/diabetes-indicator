package com.paulok;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;

import static com.paulok.DiabetesIndicatorFileStructureUtils.FEATURES;
import static com.paulok.DiabetesIndicatorFileStructureUtils.LABEL_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.PREDICTION_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.FEATURES_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.RAW_PREDICTION_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.PROBABILITY_COLUMN_NAME;

public class DiabetesPredictor {

    private static final String DATA_FILE_PATH = "diabetes_test.csv";
    private static final String MODEL_PATH = "model";

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);

        ArgumentParser argumentParser = new ArgumentParser();
        String[] adjustedArgs = argumentParser.parseArguments(args, DATA_FILE_PATH, MODEL_PATH);
        String dataFilePath = adjustedArgs[0];
        String modelPath = adjustedArgs[1];

        Dataset<Row> ds = SparkUtils.getSparkSession()
                .read()
                .format("csv")
                .option("header", "true")
                .load(dataFilePath)
                .cache();

        DatasetTuner tuner = new DatasetTuner();
        ds = tuner.prepareDatasetForClassification(ds, LABEL_COLUMN_NAME, FEATURES);

        PipelineModel model = PipelineModel.load(modelPath);
        Dataset<Row> predictions = model.transform(ds);

        AccuracyEvaluatorForBinaryClassification evaluator = new AccuracyEvaluatorForBinaryClassification();
        System.err.println("Test data accuracy: " + evaluator.getAccuracy(
                predictions, PREDICTION_COLUMN_NAME, LABEL_COLUMN_NAME));

        predictions.drop(FEATURES_COLUMN_NAME, RAW_PREDICTION_COLUMN_NAME, PROBABILITY_COLUMN_NAME)
                .coalesce(1)
                .write()
                .mode(SaveMode.Overwrite)
                .option("header", true)
                .csv("predictions");
    }
}
