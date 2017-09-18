package apps.meetpuntdetector.st4;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Detector {
    public void createTrainingData() throws Exception {
        String trainingMetaData = "resources/meetpuntdetector/good-samples";
        String goodSampleDir = "good-samples";
        String badSampleDir = "bad-samples";

        List<Sample> samples = new ArrayList<>();

        BufferedReader reader = getReader(trainingMetaData);

        String line;
        while((line = reader.readLine()) != null) {
            line = line.trim();
            if(line.length() == 0 || !Character.isDigit(line.charAt(0))) {
                continue;
            }
            samples.add(Sample.readSample(line));
        }

        St4Reader st4Reader = new St4Reader("resources/meetpuntdetector/M170829.st4");
        st4Reader.setOutputDir("good-samples");
        st4Reader.process(samples);

        List<Sample> badSamples = new ArrayList<>();

        // Create bad samples by changing one meetpunt to the other direction.
        for(Sample sample : samples) {
            char direction = sample.getDirection() == 'L' ? 'R' : 'L';
            Sample badSample = new Sample(sample.getRoadNumber(), direction, sample.getMeetPunt(), sample.getMeetPuntRange(), sample.getMinute(), sample.getMeetPuntRange());
            badSamples.add(badSample);
        }

        st4Reader = new St4Reader("resources/meetpuntdetector/M170829.st4");
        st4Reader.setOutputDir("bad-samples");
        st4Reader.process(badSamples);

        int MEET_PUNT_SWITCHED = 1;

        for(File file : new File(badSampleDir).listFiles()) {
            String [] arr = file.getName().split("-");
            String goodSample = String.format("%s/%s-%c-%s-%s", goodSampleDir, arr[0], arr[1].charAt(0) == 'L' ? 'R' : 'L', arr[2], arr[3]);

            String line2Switch = getLine(goodSample, MEET_PUNT_SWITCHED);

            reader = getReader(file.getAbsolutePath());

            FileOutputStream out = getWriter(String.format("%s-tmp", file.getAbsolutePath()));

            int lineNo=0;
            while((line = reader.readLine()) != null) {
                if(lineNo++ == MEET_PUNT_SWITCHED) {
                    out.write((line2Switch + "\n").getBytes());
                } else {
                    out.write((line + "\n").getBytes());
                }
            }
            out.close();
            reader.close();

            new File(String.format("%s-tmp", file.getAbsolutePath())).renameTo(new File(String.format("%s", file.getAbsolutePath())));
        }
    }

    private String getLine(String file, int lineNo) throws Exception {
        BufferedReader reader = getReader(file);
        String s=null;
        while(lineNo-- >= 0) {
            s = reader.readLine();
        }
        return s;
    }

    private BufferedReader getReader(String file) throws Exception {
        return new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    }

    private FileOutputStream getWriter(String file) throws Exception {
        return new FileOutputStream(file);
    }
}
