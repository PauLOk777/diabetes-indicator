package com.paulok;

public class ArgumentParser {

    private static final int ARGUMENTS_SIZE = 2;

    public String[] parseArguments(String[] args, String defaultDataFilePath, String defaultModelPath) {
        String[] adjustedArguments = new String[ARGUMENTS_SIZE];

        if (args.length > 0) {
            adjustedArguments[0] = args[0];
        } else {
            adjustedArguments[0] = defaultDataFilePath;
        }

        if (args.length > 1) {
            adjustedArguments[1] = args[1];
        } else {
            adjustedArguments[1] = defaultModelPath;
        }

        return adjustedArguments;
    }
}
