package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;
import java.util.*;

public abstract class SourceClassifier {

    List<ScalableLengthNetwork> networks = new ArrayList<>();

    public void addNetwork(ScalableLengthNetwork network) throws Exception {
        networks.add(network);
        network.readWeights();
    }

    double scaleInput(int input) {
        if (input >= 10) {
            return 0.99;
        }

        if (input == 0) {
            return 0.01;
        }

        return (double) input / 10;
    }

    private String getExtension(String file) {
        int index;
        if((index = file.lastIndexOf(".")) < 0) {
            return null;
        }
        return file.substring(index + 1, file.length());
    }

    public void learn(File root) throws Exception {

        int [] summedIterations = new int[networks.size()];

        int fileIterations=0;

        for(File file : shuffle(root.listFiles())) {

            String extenstion = getExtension(file.getName());

            boolean isJava = "java".equals(extenstion);
            boolean isPython = "py".equals(extenstion);
            boolean isC = "c".equals(extenstion) || "h".equals(extenstion);

            System.out.println(String.format("File: %s", file.getName()));

            KeywordCounter keywordCounter = new KeywordCounter(file);

            double[] inputs = new double[keywordCounter.keywordOccurrences.length];
            for (int idx = 0; idx < keywordCounter.keywordOccurrences.length; idx++) {
                inputs[idx] = scaleInput(keywordCounter.keywordOccurrences[idx]);
            }

            int maxIterations = 50000;
            int iterations;
            double error = 0.0001;

            fileIterations++;

            int networkNo = 0;

            for(ScalableLengthNetwork network : networks) {

                double [] targets = composeTargets(network, isJava, isPython, isC);

                try {
                    iterations = network.learn(inputs, targets, error, maxIterations);
                    summedIterations[networkNo] += iterations;

                    System.out.println(String.format("Network: %s, files: %d, iterations: %d, avg: %d",
                            network,
                            fileIterations,
                            iterations,
                            summedIterations[networkNo++]/fileIterations));
                    System.out.println();

                } catch (Exception e) {
                    throw new RuntimeException(String.format("Max iterations exceeded for file %s.", file.getName(), e));
                }
            }
        }

        for(ScalableLengthNetwork network : networks) {
            network.writeWeights();
        }
    }

    abstract double [] composeTargets(ScalableLengthNetwork network, boolean isJava, boolean isPython, boolean isC);

    private List<File> shuffle(File [] fileArr) {
        List<File> files = new ArrayList<>();
        for(File f : fileArr) {
            String ext = getExtension(f.getName());
            if(!ext.equals("c") && !ext.equals("java") && !ext.equals("py")) {
                continue;
            }
            files.add(f);
        }

        List<File> shuffledFiles = new ArrayList<>();

        List extensions = new ArrayList();
        int idx=0;
        do {
            extensions.clear();
            for(File file : files) {
                String ext = getExtension(file.getName());
                if(!extensions.contains(ext)) {
                    extensions.add(ext);
                }
            }

            if(idx >= extensions.size()) {
                idx = 0;
            }

            for(Iterator<File> it = files.iterator(); it.hasNext();) {
                File file = it.next();

                if(getExtension(file.getName()).equals(extensions.get(idx))) {
                    shuffledFiles.add(file);
                    it.remove();

                    if(++idx == extensions.size()) {
                        idx = 0;
                    }
                }
            }

        } while(files.size() > 0);

        return shuffledFiles;
    }
}