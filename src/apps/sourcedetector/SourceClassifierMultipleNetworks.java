package apps.sourcedetector;

import nn.Network;

import java.io.File;

public class SourceClassifierMultipleNetworks extends SourceClassifier {

    private static Network javaClassifier = new Network("javaClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 10, 1});
    private static Network pythonClassifier = new Network("pythonClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 10,  1});
    private static Network cClassifier = new Network("cClassifier", new int [] {KeywordCounter.getNoOfKeywords(), 10, 1});

    private boolean isJava, isPython, isC;

    public int recognize(String file) throws Exception {

        isJava = isPython = isC = false;

        addNetwork(javaClassifier);
        addNetwork(pythonClassifier);
        addNetwork(cClassifier);

        File unknownSource = new File(file);
        KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

        double [] inputs = new double[keywordCounter.keywordOccurrences.length];
        for(int idx=0; idx < keywordCounter.keywordOccurrences.length; idx++) {
            inputs[idx] = scaleInput(keywordCounter.keywordOccurrences[idx]);
        }

        javaClassifier.passForward(inputs);
        pythonClassifier.passForward(inputs);
        cClassifier.passForward(inputs);

        double javaOutput = javaClassifier.getOutput(0);
        double pythonOutput = pythonClassifier.getOutput(0);
        double cOutput = cClassifier.getOutput(0);

        System.out.println(String.format("Source: %s", unknownSource.getName()));

        if(javaOutput > 0.9) {
            isJava = true;
            return 1;
        }

        if(pythonOutput > 0.9) {
            isPython = true;
            return 1;
        }

        if(cOutput > 0.9) {
            isC = true;
            return 1;
        }

        return 0;
    }

    public void learn(String dir) throws Exception {
        addNetwork(javaClassifier);
        addNetwork(pythonClassifier);
        addNetwork(cClassifier);
        learn(new File(dir));
    }

    double [] composeTargets(Network network, boolean isJava, boolean isPython, boolean isC) {
        double target=0.01;

        if(network.toString().equals("javaClassifier") && isJava) {
            target = 0.99;
        }

        if(network.toString().equals("pythonClassifier") && isPython) {
            target = 0.99;
        }

        if(network.toString().equals("cClassifier") && isC) {
            target = 0.99;
        }

        return new double[]{target};
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

