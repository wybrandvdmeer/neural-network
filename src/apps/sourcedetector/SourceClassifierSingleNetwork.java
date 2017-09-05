package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;

public class SourceClassifierSingleNetwork extends SourceClassifier {

    private static ScalableLengthNetwork network = new ScalableLengthNetwork("sourceClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 40, 20, 3});

    public static int main(String [] args) throws Exception {

        SourceClassifierSingleNetwork sourceClassifier = new SourceClassifierSingleNetwork();
        sourceClassifier.addNetwork(network);

        if (args[0].equals("recognize")) {
            File unknownSource = new File(args[1]);
            KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

            double[] inputs = new double[keywordCounter.keywordOccurrences.length];
            for (int idx = 0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                inputs[idx] = sourceClassifier.scaleInput(keywordCounter.keywordOccurrences[idx]);
            }

            network.passForward(inputs);

            double javaOutput = network.getOutput(0);
            double pythonOutput = network.getOutput(1);
            double cOutput = network.getOutput(2);

            System.out.println(String.format("Source: %s", unknownSource.getName()));

            if (javaOutput > 0.9) {
                System.out.println("Source is recognized as a Java file.");
                return 1;
            } else {
                System.out.println("Source is not recognized as a Java file.");
            }

            if (pythonOutput > 0.9) {
                System.out.println("Source is recognized as a Python file.");
                return 1;
            } else {
                System.out.println("Source is not recognized as a Python file.");
            }

            if (cOutput > 0.9) {
                System.out.println("Source is recognized as a C file.");
                return 1;
            } else {
                System.out.println("Source is not recognized as a C file.");
            }
        } else if (args[0].equals("learn")) {
            sourceClassifier.learn(new File(args[1]));
        }
        return 0;
    }

    double [] composeTargets(ScalableLengthNetwork network, boolean isJava, boolean isPython, boolean isC) {
        double [] targets = new double[3];
        if(isJava) {
            targets[0] = 0.99;
            targets[1] = 0.01;
            targets[2] = 0.01;
        } else if(isPython) {
            targets[0] = 0.01;
            targets[1] = 0.99;
            targets[2] = 0.01;
        } else if(isC) {
            targets[0] = 0.01;
            targets[1] = 0.01;
            targets[2] = 0.99;
        }

        return targets;
    }
}

