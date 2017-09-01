package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Arrays;

public class SourceClassifier {

    private static ScalableLengthNetwork javaClassifier = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 10, 1});
    private static ScalableLengthNetwork pythonClassifier = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 10, 1});

    public static void main(String [] args) throws Exception {

        javaClassifier.setName("javaClassifier");
        pythonClassifier.setName("pythonClassifier");

        File javaWeights = new File("javaClassifier");
        File pythonWeights = new File("pythonClassifier");

        if(javaWeights.exists() && pythonWeights.exists()) {
            javaClassifier.read(new FileInputStream(javaWeights));
            pythonClassifier.read(new FileInputStream(pythonWeights));

            File unknownSource = new File(args[0]);
            KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

            double [] inputs = new double[keywordCounter.keywordOccurrences.length];
            for(int idx=0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                inputs[idx] = (double)keywordCounter.keywordOccurrences[idx];
            }

            javaClassifier.passForward(inputs);
            pythonClassifier.passForward(inputs);

            double javaOutput = javaClassifier.getOutput(0);
            double pythonOutput = pythonClassifier.getOutput(0);

            if(javaOutput > 0.9) {
                System.out.println("Source is recognized as a Java file.");
            } else {
                System.out.println("Source is not recognized as a Java file.");
            }

            if(pythonOutput > 0.9) {
                System.out.println("Source is recognized as a Python file.");
            } else {
                System.out.println("Source is not recognized as a Python file.");
            }

            return;
        }

        Arrays.stream(args).forEach(arg -> {
            File sourceDir = new File(arg);

            if(!sourceDir.isDirectory()) {
                throw new RuntimeException(String.format("Argument %s is not a directory.", args[0]));
            }

            try {
                processFiles(sourceDir);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        File weights = new File("javaClassifier");
        javaClassifier.write(new FileOutputStream(weights));

        weights = new File("pythonClassifier");
        pythonClassifier.write(new FileOutputStream(weights));
    }

    private static void processFiles(File root) throws Exception {
        for(File file : root.listFiles()) {
            if(file.isDirectory()) {
                processFiles(file);
            } else {
                String extenstion = getExtension(file.getName());

                boolean isJava = "java".equals(extenstion);
                boolean isPython = "py".equals(extenstion);

                System.out.println(String.format("File: %s", file.getName()));

                KeywordCounter keywordCounter = new KeywordCounter(file);

                double [] inputs = new double[keywordCounter.keywordOccurrences.length];
                for(int idx=0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                    inputs[idx] = (double)keywordCounter.keywordOccurrences[idx];
                }

                javaClassifier.learn(inputs, isJava ? new double[] {0.99} : new double[] {0.01}, 0.0001);
                pythonClassifier.learn(inputs, isPython ? new double[] {0.99} : new double[] {0.01}, 0.0001);
            }
        }
    }

    private static String getExtension(String file) {
        int index;
        if((index = file.lastIndexOf(".")) < 0) {
            return null;
        }
        return file.substring(index + 1, file.length());
    }
}

