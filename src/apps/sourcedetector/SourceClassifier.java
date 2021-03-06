package apps.sourcedetector;

import nn.Network;

import java.io.File;
import java.util.*;

public abstract class SourceClassifier {

    List<Network> networks = new ArrayList<>();

    public void addNetwork(Network network) throws Exception {
        networks.add(network);
        network.read();
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

        int [][] summedIterations = new int[networks.size()][10];

        int fileNo=0;
        int noOfIterations = 0;

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
            double error = 0.000001;
            fileNo++;

            int networkNo = 0;

            for(Network network : networks) {

                double [] targets = composeTargets(network, isJava, isPython, isC);

                try {
                    iterations = network.learn(inputs, targets, error, maxIterations);
                    if(iterations == -1) {
                        throw new RuntimeException("Max iterations exceeded for classifier");
                    }

                    if(noOfIterations < 10) {
                        summedIterations[networkNo][noOfIterations++] = iterations;
                    } else {
                        System.arraycopy(summedIterations[networkNo], 1, summedIterations[networkNo], 0, 9);
                        summedIterations[networkNo][9] = iterations;
                    }

                    int totalNoOfIterations = 0;
                    for(int idx=0; idx < noOfIterations; idx++) {
                        totalNoOfIterations += summedIterations[networkNo][idx];
                    }

                    System.out.println(String.format("Network: %s, files: %d, iterations: %d, avg: %d",
                            network,
                            fileNo,
                            iterations,
                            totalNoOfIterations/noOfIterations));
                } catch (Exception e) {
                    throw new RuntimeException(String.format("Max iterations exceeded for file %s.", file.getName(), e));
                }
            }

            System.out.println();
        }

        for(Network network : networks) {
            network.write();
        }
    }

    abstract double [] composeTargets(Network network, boolean isJava, boolean isPython, boolean isC);

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