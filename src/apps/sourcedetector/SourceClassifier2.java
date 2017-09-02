package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.*;

public class SourceClassifier2 {

    private static ScalableLengthNetwork sourceClassifier = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 40, 20, 3});

    public static void main(String [] args) throws Exception {

        sourceClassifier.setName("sourceClassifier");

        File weights = new File("sourceClassifier");

        if(weights.exists()) {
            sourceClassifier.read(new FileInputStream(weights));

            File unknownSource = new File(args[0]);
            KeywordCounter keywordCounter = new KeywordCounter(unknownSource);

            double [] inputs = new double[keywordCounter.keywordOccurrences.length];
            for(int idx=0; idx < keywordCounter.keywordOccurrences.length; idx++) {
               inputs[idx] = scaleInput(keywordCounter.keywordOccurrences[idx]);
            }

            sourceClassifier.passForward(inputs);

            double javaOutput = sourceClassifier.getOutput(0);
            double pythonOutput = sourceClassifier.getOutput(1);
            double cOutput = sourceClassifier.getOutput(2);

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
                processFiles(sourceDir);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        weights = new File("sourceClassifier");
        sourceClassifier.write(new FileOutputStream(weights));
    }

    private static double scaleInput(int input) {
        if(input >= 10) {
            return 0.99;
        }

        if(input == 0) {
            return 0.01;
        }

        return (double)input / 10;
    }

    private static int fileIterations=0;
    private static int summedIterations=0;

    private static void processFiles(File root) throws Exception {
        for(File file : shuffle(root.listFiles())) {
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
                    inputs[idx] = scaleInput(keywordCounter.keywordOccurrences[idx]);
                }

                int maxIterations = 100000;
                int iterations;
                double error = 0.000001;

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
                } else {
                    throw new RuntimeException("Unknown source.");
                }

                try {
                    iterations = sourceClassifier.learn(inputs, targets, error, maxIterations);
                    summedIterations += iterations;

                    System.out.println(String.format("Files: %d, iterations: %d, avg: %d",
                            fileIterations,
                            iterations,
                            summedIterations / ++fileIterations));
                    System.out.println();

                } catch(Exception e) {
                    throw new RuntimeException(String.format("Max iterations exceeded for file %s.", file.getName(), e));
                }
            }
        }
    }

    private static List<File> shuffle(File [] fileArr) {
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

    private static String getExtension(String file) {
        int index;
        if((index = file.lastIndexOf(".")) < 0) {
            return null;
        }
        return file.substring(index + 1, file.length());
    }
}

