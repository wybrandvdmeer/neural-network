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
        sourceClassifierSingleNetwork.main(new String[] {"learn", "/home/wybrand/source"});
    }

    @Test
    public void testRecoginizeSingleNetwork() throws Exception {
        SourceClassifierSingleNetwork sourceClassifierSingleNetwork = new SourceClassifierSingleNetwork();

        File test = new File("/home/wybrand/test");

        int files=0;
        int recognized=0;

        for(File file : test.listFiles()) {
            recognized += sourceClassifierSingleNetwork.main(new String[]{"recognize", file.getAbsolutePath()});
            files++;
        }

        System.out.println(String.format("Recognized %f percent\n", (double)recognized/files));
    }
}
