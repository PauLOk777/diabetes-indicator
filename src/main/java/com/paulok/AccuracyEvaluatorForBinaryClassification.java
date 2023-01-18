package com.paulok;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static com.paulok.DiabetesIndicatorFileStructureUtils.LABEL_COLUMN_NAME;
import static com.paulok.DiabetesIndicatorFileStructureUtils.PREDICTION_COLUMN_NAME;

public class AccuracyEvaluatorForBinaryClassification {

    public double getAccuracy(Dataset<Row> predictedData, String predictedColumnName, String labelColumnName) {
        return predictedData.filter(PREDICTION_COLUMN_NAME + "==" + LABEL_COLUMN_NAME).count() /
                (double) predictedData.count();
    }
}
