package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class SourceClassifier {

    private static ScalableLengthNetwork javaClassifier = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 10, 5, 1});
    private static ScalableLengthNetwork pythonClassifier = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 10, 5, 1});
    private static ScalableLengthNetwork cClassifier = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 10, 5, 1});

    public static void main(String [] args) throws Exception {

        javaClassifier.setName("javaClassifier");
        pythonClassifier.setName("pythonClassifier");
        cClassifier.setName("cClassifier");

        File javaWeights = new File("javaClassifier");
        File pythonWeights = new File("pythonClassifier");
        File cWeights = new File("cClassifier");

        if(javaWeights.exists() && pythonWeights.exists() && cWeights.exists()) {
            javaClassifier.read(new FileInputStream(javaWeights));
            pythonClassifier.read(new FileInputStream(pythonWeights));
            cClassifier.read(new FileInputStream(cWeights));

            File unknownSource = new File(args[0]);
            KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

            double [] inputs = new double[keywordCounter.keywordOccurrences.length];
            for(int idx=0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                inputs[idx] = (double)keywordCounter.keywordOccurrences[idx];
            }

            javaClassifier.passForward(inputs);
            pythonClassifier.passForward(inputs);
            cClassifier.passForward(inputs);

            double javaOutput = javaClassifier.getOutput(0);
            double pythonOutput = pythonClassifier.getOutput(0);
            double cOutput = cClassifier.getOutput(0);

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

            if(cOutput > 0.9) {
                System.out.println("Source is recognized as a C file.");
            } else {
                System.out.println("Source is not recognized as a C file.");
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

        weights = new File("cClassifier");
        cClassifier.write(new FileOutputStream(weights));
    }

    private static void processFiles(File root) throws Exception {
        List<File> files = Arrays.asList(root.listFiles());
        Collections.shuffle(files);
        for(File file : files) {
            if(file.isDirectory()) {
                processFiles(file);
            } else {
                String extenstion = getExtension(file.getName());

                boolean isJava = "java".equals(extenstion);
                boolean isPython = "py".equals(extenstion);
                boolean isC = "c".equals(extenstion);

                System.out.println(String.format("File: %s", file.getName()));

                KeywordCounter keywordCounter = new KeywordCounter(file);

                double[] inputs = new double[keywordCounter.keywordOccurrences.length];
                for (int idx = 0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                    inputs[idx] = (double) keywordCounter.keywordOccurrences[idx];
                }

                int maxIterations = 50000;

                try {
                    pythonClassifier.learn(inputs, isPython ? new double[]{0.99} : new double[]{0.01}, 0.0001, maxIterations);
                    cClassifier.learn(inputs, isC ? new double[]{0.99} : new double[]{0.01}, 0.0001, maxIterations);
                    javaClassifier.learn(inputs, isJava ? new double[]{0.99} : new double[]{0.01}, 0.0001, maxIterations);
                } catch(Exception e) {
                    throw new RuntimeException(String.format("Max iterations exceeded for file %s.", file.getName(), e));
                }
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

