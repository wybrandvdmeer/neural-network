package neuralnetwork;

import apps.sourcedetector.SourceClassifierMultipleNetworks;
import apps.sourcedetector.SourceClassifierSingleNetwork;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TestSourceRecognizer {

    @Test
    public void testLearnMultipleNetworks() throws Exception {
        SourceClassifierMultipleNetworks sourceClassifierMultipleNetworks = new SourceClassifierMultipleNetworks();
        sourceClassifierMultipleNetworks.main(new String[] {"learn", "/home/wybrand/source"});
    }

    @Test
    public void testRecoginizeMultipleNetworks() throws Exception {
        SourceClassifierMultipleNetworks sourceClassifierMultipleNetworks = new SourceClassifierMultipleNetworks();

        File test = new File("/home/wybrand/test");

        int files=0;
        int recognized=0;

        for(File file : test.listFiles()) {
            recognized += sourceClassifierMultipleNetworks.main(new String[]{"recognize", file.getAbsolutePath()});
            files++;
        }

        System.out.println(String.format("Recognized %f percent\n", (double)recognized/files));
    }

    @Test
    public void testLearnSingleNetworks() throws Exception {
        SourceClassifierSingleNetwork sourceClassifierSingleNetwork = new SourceClassifierSingleNetwork();
        sourceClassifierSingleNetwork.learn("/home/wybrand/source");
    }

    @Test
    public void testRecoginizeSingleNetwork() throws Exception {
        SourceClassifierSingleNetwork sourceClassifierSingleNetwork = new SourceClassifierSingleNetwork();

        File test = new File("/home/wybrand/test");

        int javaFiles=0;
        int pythonFiles=0;
        int cFiles=0;
        int recognized=0;
        int files=0;
        int recognizedAsJava=0, recognizedAsPython=0, recognizedAsC=0;
        List<String> recognizeErrors = new ArrayList<>();

        for(File file : test.listFiles()) {
            String ext = getExtension(file.getName());

            recognized += sourceClassifierSingleNetwork.recognize(file.getAbsolutePath());
            files++;

            boolean isJava = sourceClassifierSingleNetwork.isJava();
            boolean isPython = sourceClassifierSingleNetwork.isPython();
            boolean isC = sourceClassifierSingleNetwork.isC();

            if(ext.equals("java")) {
                javaFiles++;
                if(isJava) {
                    recognizedAsJava++;
                } else if(isPython || isC){
                    recognizeErrors.add(String.format("%s is recognized as %s.", file.getName(), isPython ? "python" : "C"));
                }
            } else
            if(ext.equals("py")) {
                pythonFiles++;
                if(isPython) {
                    recognizedAsPython++;
                } else if(isJava || isC){
                    recognizeErrors.add(String.format("%s is recognized as %s.", file.getName(), isJava ? "java" : "C"));
                }
            } else
            if(ext.equals("c")) {
                cFiles++;
                if(isC) {
                    recognizedAsC++;
                } else if(isPython || isJava){
                    recognizeErrors.add(String.format("%s is recognized as %s.", file.getName(), isPython ? "python" : "java"));
                }
            }

            System.out.println(String.format("Recognized: %d, java: %s, python: %s, C: %s", recognized,
                    sourceClassifierSingleNetwork.isJava(),
                    sourceClassifierSingleNetwork.isPython(),
                    sourceClassifierSingleNetwork.isC()));
        }

        System.out.println(String.format("Java recognized %f percent\n", (double)recognizedAsJava/javaFiles));
        System.out.println(String.format("Python recognized %f percent\n", (double)recognizedAsPython/pythonFiles));
        System.out.println(String.format("C recognized %f percent\n", (double)recognizedAsC/cFiles));
        System.out.println(String.format("Recognized %f percent\n", (double)recognized/files));

        System.out.println("Recognize errors: " + recognizeErrors.size());
        for(String s : recognizeErrors) {
            System.out.println(s);
        }
    }

    private String getExtension(String file) {
        int index;
        if((index = file.lastIndexOf(".")) < 0) {
            return null;
        }
        return file.substring(index + 1, file.length());
    }
}
