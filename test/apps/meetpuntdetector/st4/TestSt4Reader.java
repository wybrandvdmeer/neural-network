package apps.meetpuntdetector.st4;

import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TestSt4Reader {
    @Test
    public void testSt4Reader() throws Exception {
        St4Reader st4Reader = new St4Reader("resources/meetpuntdetector/M170829.st4");

        Sample a1R = new Sample(1, 'R', 0, 2, 0, 2);

        List<Sample> samples = new ArrayList<>();
        samples.add(a1R);

        st4Reader.process(samples);
    }

    @Test
    public void testLearner() throws Exception {
        Detector detector = new Detector();
        detector.createTrainingData();
    }
}
