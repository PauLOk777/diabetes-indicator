package com.paulok;

public class DiabetesIndicatorFileStructureUtils {

    public static final String FEATURES_DIVIDED_BY_COMMA = "HighBP,HighChol,CholCheck,BMI,Smoker,Stroke," +
            "HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth," +
            "MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income";
    public static final String[] FEATURES = FEATURES_DIVIDED_BY_COMMA.split(",");
    public static final String LABEL_COLUMN_NAME = "Diabetes";
    public static final String PREDICTION_COLUMN_NAME = "prediction";
    public static final String FEATURES_COLUMN_NAME = "features";
    public static final String RAW_PREDICTION_COLUMN_NAME = "rawPrediction";
    public static final String PROBABILITY_COLUMN_NAME = "probability";
}
