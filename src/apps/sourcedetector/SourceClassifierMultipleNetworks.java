package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;

public class SourceClassifierMultipleNetworks extends SourceClassifier {

    private static ScalableLengthNetwork javaClassifier = new ScalableLengthNetwork("javaClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 20, 10, 1});
    private static ScalableLengthNetwork pythonClassifier = new ScalableLengthNetwork("pythonClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 20, 10,  1});
    private static ScalableLengthNetwork cClassifier = new ScalableLengthNetwork("cClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 20, 10, 1});

    public static void main(String [] args) throws Exception {

        SourceClassifierMultipleNetworks sourceClassifier = new SourceClassifierMultipleNetworks();
        sourceClassifier.addNetwork(javaClassifier);
        sourceClassifier.addNetwork(pythonClassifier);
        sourceClassifier.addNetwork(cClassifier);

        if(args[0].equals("recognize")) {
            File unknownSource = new File(args[1]);
            KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

            double [] inputs = new double[keywordCounter.keywordOccurrences.length];
            for(int idx=0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                inputs[idx] = sourceClassifier.scaleInput(keywordCounter.keywordOccurrences[idx]);
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
        } else if(args[0].equals("learn")) {
            sourceClassifier.learn(new File(args[1]));
        }
    }
}

