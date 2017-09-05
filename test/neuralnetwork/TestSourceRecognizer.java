package neuralnetwork;

import apps.sourcedetector.SourceClassifierMultipleNetworks;
import apps.sourcedetector.SourceClassifierSingleNetwork;
import org.junit.Test;

import java.io.File;

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

        for(File file : test.listFiles()) {
            String ext = getExtension(file.getName());

            recognized += sourceClassifierSingleNetwork.recognize(file.getAbsolutePath());
            files++;

            if(ext.equals("java")) {
                javaFiles++;
                if(sourceClassifierSingleNetwork.isJava()) {
                    recognizedAsJava++;
                }
            } else
            if(ext.equals("py")) {
                pythonFiles++;
                if(sourceClassifierSingleNetwork.isPython()) {
                    recognizedAsPython++;
                }
            } else
            if(ext.equals("c")) {
                cFiles++;
                if(sourceClassifierSingleNetwork.isJava()) {
                    recognizedAsC++;
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
    }

    private String getExtension(String file) {
        int index;
        if((index = file.lastIndexOf(".")) < 0) {
            return null;
        }
        return file.substring(index + 1, file.length());
    }
}
