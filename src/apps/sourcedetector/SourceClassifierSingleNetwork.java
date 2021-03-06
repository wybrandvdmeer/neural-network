package apps.sourcedetector;

import nn.Network;

import java.io.File;

public class SourceClassifierSingleNetwork extends SourceClassifier {

    private Network network = new Network("sourceClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 40, 20, 3});

    private boolean isJava, isPython, isC;

    public int recognize(String file) throws Exception {

        isJava = isPython = isC = false;

        addNetwork(network);

        File unknownSource = new File(file);
        KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

        double[] inputs = new double[keywordCounter.keywordOccurrences.length];
        for (int idx = 0; idx < keywordCounter.keywordOccurrences.length; idx++) {
            inputs[idx] = scaleInput(keywordCounter.keywordOccurrences[idx]);
        }

        network.passForward(inputs);

        double javaOutput = network.getOutput(0);
        double pythonOutput = network.getOutput(1);
        double cOutput = network.getOutput(2);

        System.out.println(String.format("Source: %s", unknownSource.getName()));

        if (javaOutput > 0.9) {
            isJava = true;
            return 1;
        }

        if (pythonOutput > 0.9) {
            isPython = true;
            return 1;
        }

        if (cOutput > 0.9) {
            isC = true;
            return 1;
        }
        return 0;
    }

    public void learn(String dir) throws Exception {
        addNetwork(network);
        learn(new File(dir));
    }

    double [] composeTargets(Network network, boolean isJava, boolean isPython, boolean isC) {
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

    public boolean isJava() {
        return isJava;
    }

    public boolean isPython() {
        return isPython;
    }

    public boolean isC() {
        return isC;
    }
}

