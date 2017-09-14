package apps.meetpuntdetector.st4;

import org.junit.Test;

import java.io.File;

public class TestSt4Reader {
    @Test
    public void testSt4Reader() throws Exception {
        St4Reader st4Reader = new St4Reader(new File("resources/meetpuntdetector/M170829.st4"));
    }
}
