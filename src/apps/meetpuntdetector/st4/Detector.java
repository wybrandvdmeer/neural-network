package apps.meetpuntdetector.st4;

import neuralnetwork.Network;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Detector {
    private String goodSampleDir = "good-samples";
    private String badSampleDir = "bad-samples";
    private double ERROR = 0.0001;

    private Network network = new Network("meetpunt-detector", new int [] {3, 5, 1});

    public void learn() throws Exception {
        File [] goodSamples = new File(goodSampleDir).listFiles();
        File [] badSamples = new File(badSampleDir).listFiles();

        for(int idx=0; idx < goodSamples.length; idx++) {
            network.learn(scaleInput(goodSamples[idx]), new double[] {0.99}, ERROR, 5000);
            network.learn(scaleInput(badSamples[idx]), new double[] {0.01}, ERROR, 5000);
        }
    }

    private double [] scaleInput(File input) throws Exception {
        List<Double> targets = new ArrayList<>();
        BufferedReader reader = getReader(input);

        String line;
        while((line = reader.readLine()) != null) {
            String [] arr = line.split(" ");
            double intensitiyA = Double.parseDouble(arr[0]);
            double intensitiyB = Double.parseDouble(arr[1]);
            double velocityA = Double.parseDouble(arr[2]);
            double velocityB = Double.parseDouble(arr[3]);

            targets.add(velocityA  + velocityB);
        }

        return targets.stream().mapToDouble(d->d).toArray();
    }

    public void createTrainingData() throws Exception {
        String trainingMetaData = "resources/meetpuntdetector/good-samples";


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
            if(line2Switch == null) {
                throw new RuntimeException(String.format("No corresponding line found for file %s.", file.getName()));
            }

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

    private BufferedReader getReader(File file) throws Exception {
        return new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    }

    private BufferedReader getReader(String file) throws Exception {
        return new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    }

    private FileOutputStream getWriter(String file) throws Exception {
        return new FileOutputStream(file);
    }
}
