package com.paulok;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.io.IOException;

import static com.paulok.DiabetesIndicatorFileStructureUtils.FEATURES;
import static com.paulok.DiabetesIndicatorFileStructureUtils.LABEL_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.PREDICTION_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.FEATURES_COLUMN_NAME;

public class ModelTraining {

    private static final String DATA_FILE_PATH = "diabetes_train.csv";
    private static final String MODEL_PATH = "model";

    public static void main(String[] args) throws IOException {
        Logger.getLogger("org").setLevel(Level.ERROR);

        ArgumentParser argumentParser = new ArgumentParser();
        String[] adjustedArgs = argumentParser.parseArguments(args, DATA_FILE_PATH, MODEL_PATH);
        String dataFilePath = adjustedArgs[0];
        String modelPath = adjustedArgs[1];

        Dataset<Row> ds = SparkUtils.readCsvFileWithHeaders(dataFilePath);

        DatasetTuner tuner = new DatasetTuner();
        ds = tuner.prepareDatasetForClassification(ds, LABEL_COLUMN_NAME, FEATURES);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(FEATURES)
                .setOutputCol("features");

        Dataset<Row>[] splits = ds.randomSplit(new double[] { 0.9, 0.1 });
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> crossValidationData = splits[1];

        RandomForestClassifier gbtClassifier = new RandomForestClassifier()
                .setLabelCol(LABEL_COLUMN_NAME)
                .setFeaturesCol(FEATURES_COLUMN_NAME)
                .setNumTrees(100)
                .setMaxDepth(12)
                .setMaxBins(50);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, gbtClassifier});

        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictionsForTrainingData = model.transform(trainingData);
        Dataset<Row> predictionsForCrossValidationData = model.transform(crossValidationData);

        AccuracyEvaluatorForBinaryClassification evaluator = new AccuracyEvaluatorForBinaryClassification();
        System.err.println("Training data accuracy: " + evaluator.getAccuracy(
                predictionsForTrainingData, PREDICTION_COLUMN_NAME, LABEL_COLUMN_NAME));
        System.err.println("Cross-validation data accuracy: " + evaluator.getAccuracy(
                predictionsForCrossValidationData, PREDICTION_COLUMN_NAME, LABEL_COLUMN_NAME));

        model.write().overwrite().save(modelPath);
    }
}
