package apps.sourcedetector;

import neuralnetwork.ScalableLengthNetwork;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Arrays;

public class SourceClassifier {

    private static ScalableLengthNetwork network = new ScalableLengthNetwork(new int [] {KeywordCounter.getNoOfKeywords(), 20, 2});

    public static void main(String [] args) throws Exception {

        Arrays.stream(args).forEach( arg -> {
            File sourceDir = new File(args[0]);

            if(!sourceDir.isDirectory()) {
                throw new RuntimeException(String.format("Argument %s is not a directory.", args[0]));
            }

            try {
                processFiles(sourceDir);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        File weights = new File("weights");
        network.write(new FileOutputStream(weights));
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

                network.learn(inputs, isJava ? new double[] {0.99, 0.01} : new double[] {0.01, 0.99}, 0.0001);
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

