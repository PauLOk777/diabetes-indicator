package com.paulok;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class DatasetTuner {

    public Dataset<Row> prepareDatasetForClassification(Dataset<Row> ds, String labelColumnName, String[] features) {
        Dataset<Row> datasetWithoutMissingValues = ds.na().drop();

        datasetWithoutMissingValues = datasetWithoutMissingValues
                .withColumn(labelColumnName, datasetWithoutMissingValues.col(labelColumnName).cast("double"));
        for (String feature: features) {
            datasetWithoutMissingValues = datasetWithoutMissingValues
                    .withColumn(feature, datasetWithoutMissingValues.col(feature).cast("double"));
        }

        return datasetWithoutMissingValues;
    }
}
