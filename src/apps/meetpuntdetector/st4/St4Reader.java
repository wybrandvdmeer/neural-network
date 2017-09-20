package apps.meetpuntdetector.st4;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.*;

public class St4Reader {

    private final FileInputStream stream;
    private String outputDir=null;

    public St4Reader(String st4) throws Exception {
        stream = new FileInputStream(new File(st4));
    }

    public void process(List<Sample> samplesArg) throws Exception {
        List<Sample> samples = new ArrayList<>(samplesArg);

        Map<Integer, List<Sample>> meetPuntID2Samples = new HashMap<>();

        long noOfMeetPunten = readHeader(stream);

        log("Found %d meetpunten.", noOfMeetPunten);

        int roadNumber=0, relativeMeetpuntID=0;
        char direction=0x00;

        for(int meetPuntID=0; meetPuntID < noOfMeetPunten; meetPuntID++) {
            MeetPunt meetPunt = readMeetPunt(stream);

            if(!meetPunt.isMainRoad() || meetPunt.isSuspect()) {
                continue;
            }

            if(roadNumber != meetPunt.getRoadNumber() || direction != meetPunt.getDirection()) {
                roadNumber = meetPunt.getRoadNumber();
                direction = meetPunt.getDirection();
                relativeMeetpuntID = 0;
            }

            System.out.println(String.format("Meetpunt: %s - %d", meetPunt, relativeMeetpuntID));

            for(Iterator<Sample> it = samples.iterator(); it.hasNext();) {
                Sample sample = it.next();
                if(sample.getRoadNumber() == meetPunt.getRoadNumber() && sample.getDirection() == meetPunt.getDirection() && sample.getMeetPunt() == relativeMeetpuntID) {
                    log("Found sampled meetpunt %s.", sample);
                    if(meetPuntID2Samples.get(meetPuntID) == null) {
                        meetPuntID2Samples.put(meetPuntID, new ArrayList<>());
                    }
                    meetPuntID2Samples.get(meetPuntID).add(sample);
                    it.remove();
                }
            }

            relativeMeetpuntID++;
        }

        if(!samples.isEmpty()) {
            System.out.println("Geen meetpunten gevonden voor volgende collectSamples:");
            for(Sample sample : samples) {
                System.out.println("Sample: " + sample);
            }
            throw new RuntimeException("Geen meetpunten gevonden voor een aantal collectSamples.");
        }

        List<Sample> samplesPerMeetpunt=null;
        Map<Sample, List<DataRecord>> dataRecordsPerSample = new HashMap<>();

        Map<Sample, List<String>> outputtedSamples = new HashMap<>();

        relativeMeetpuntID = 0;

        /* Read the data. Data is in order of the meetPunten, and for each Meetpunt is starts from minute 0 to minute 1439.
        */
        for(int meetPuntID=0; meetPuntID < noOfMeetPunten; meetPuntID++) {
            if(samplesPerMeetpunt == null) {
                outputtedSamples.clear();

                for (int key : meetPuntID2Samples.keySet()) {
                    if (key == meetPuntID) {
                        samplesPerMeetpunt = meetPuntID2Samples.get(key);
                        samplesPerMeetpunt.forEach(sample -> outputtedSamples.put(sample, new ArrayList<>()));
                        relativeMeetpuntID = 0;

                        dataRecordsPerSample.clear();
                        samplesPerMeetpunt.forEach(sample -> dataRecordsPerSample.put(sample, new ArrayList<>()));
                        break;
                    }
                }
            }

            for(int minute=0; minute < 1440; minute++) {
                DataRecord dataRecord = readDataRecord(stream, minute);

                for(int idx=0; samplesPerMeetpunt != null && idx < samplesPerMeetpunt.size(); idx++) {
                    Sample sample = samplesPerMeetpunt.get(idx);
                    if(sample.meetPuntInRange(relativeMeetpuntID) && sample.minuteInRange(minute)) {
                        dataRecordsPerSample.get(sample).add(dataRecord);
                    }
                }
            }

            if(samplesPerMeetpunt != null) {

                for(Iterator<Sample> it = samplesPerMeetpunt.iterator(); it.hasNext();) {
                    Sample sample = it.next();

                    if(sample.meetPuntInRange(relativeMeetpuntID)) {
                        collectSamples(outputtedSamples.get(sample), sample.getMinutesInRange(), dataRecordsPerSample.get(sample));
                    }

                    if(sample.meetPuntAboveRange(relativeMeetpuntID)) {
                        writeSamples(outputtedSamples.get(sample), sample);
                        it.remove();
                    }
                }

                if(samplesPerMeetpunt.isEmpty()) {
                    meetPuntID2Samples.values().remove(samplesPerMeetpunt);
                    samplesPerMeetpunt = null;
                }

                if(meetPuntID2Samples.isEmpty()) {
                    break;
                }

                relativeMeetpuntID++;
            }
        }
    }

    private void writeSamples(List<String> samples, Sample sample) throws Exception {
        FileOutputStream out;
        if(outputDir != null) {
            out = new FileOutputStream(new File(String.format("%s/%d-%c-%d-%d", outputDir, sample.getRoadNumber(), sample.getDirection(), sample.getMeetPunt(), sample.getMinute())));
        } else {
            out = new FileOutputStream(new File(String.format("%d-%c-%d-%d", sample.getRoadNumber(), sample.getDirection(), sample.getMeetPunt(), sample.getMinute())));
        }

        for(String s : samples) {
            out.write(s.getBytes());
        }
        out.close();
    }

    private void collectSamples(List<String> sampleFile, int minuntes, List<DataRecord> dataRecords) {

        int intensityA=0, intentityB=0, velocityA=0, velocityB=0;

        for(DataRecord dataRecord : dataRecords) {
            intensityA += dataRecord.getIntensityA();
            intentityB += dataRecord.getIntensityB();
            velocityA += dataRecord.getVelocityA();
            velocityB += dataRecord.getVelocityB();
        }

        sampleFile.add(String.format("%.2f %.2f %.2f %.2f\n", (double)intensityA/minuntes,(double)intentityB/minuntes, (double)velocityA/minuntes, (double)velocityB/minuntes));
    }

    private void log(String format, Object... args) {
        System.out.println(String.format(format, args));
    }

    private DataRecord readDataRecord(FileInputStream fileInputStream, int minute) throws Exception {
        DataRecord dataRecord = new DataRecord(minute);
        dataRecord.setIntensityA(readInt(fileInputStream, 1));
        dataRecord.setIntensityB(readInt(fileInputStream, 1));
        dataRecord.setVelocityA(readInt(fileInputStream, 1));
        dataRecord.setVelocityB(readInt(fileInputStream, 1));

        /* Not used (by us).
        */
        byte [] bytes = new byte[4];
        fileInputStream.read(bytes);

        return dataRecord;
    }

    private MeetPunt readMeetPunt(FileInputStream fileInputStream) throws Exception {

        long measurmentNo = readInt(fileInputStream, 2);

        MeetPunt meetPunt = new MeetPunt(measurmentNo);

        int meetType = readInt(fileInputStream, 1); // 'meetType' which is not used.

        meetPunt.setRoadName(readString(fileInputStream, 7));
        meetPunt.setMeterPosition(readLong(fileInputStream));
        meetPunt.setDistanceToDS(readInt(fileInputStream, 2));
        meetPunt.setNoOfLanesA(readInt(fileInputStream, 1));
        meetPunt.setNoOfLanesB(readInt(fileInputStream, 1));
        meetPunt.setOSType(readInt(fileInputStream, 1));
        meetPunt.setGP(readInt(fileInputStream, 1));
        meetPunt.setLaneData(readInt(fileInputStream, 1));
        meetPunt.setShowMap(readLong(fileInputStream));
        meetPunt.setBlockMap(readLong(fileInputStream));
        readLong(fileInputStream); // reserved

        return meetPunt;
    }

    private long readHeader(FileInputStream fileInputStream) throws Exception {
        long noOfMeetPunten = readLong(fileInputStream);
        long version = readLong(fileInputStream);
        long unUsedHeaderSpace = readLong(fileInputStream);

        for(int idx=0; idx < unUsedHeaderSpace; idx++) {
            int i = readInt(fileInputStream, 1);
        }

        return noOfMeetPunten;
    }

    private String readString(FileInputStream fileInputStream, int size) throws Exception {
        byte [] bytes = new byte[size];
        for(int idx=0; idx < size; idx++) {
            int token = fileInputStream.read();
            if(token == -1) {
                throw new RuntimeException("Unexpected EOF");
            }
            bytes[idx] = (byte)token;
        }
        return new String(bytes);
    }

    /**
     * Function reads an unsigned Big Endian integer of 4 bytes.
     */
    private int readInt(FileInputStream fileInputStream, int size) throws Exception {

        if(size >= 4) {
            throw new RuntimeException("Unsigned integer is less than 4 bytes.");
        }

        byte [] bytes = new byte[size];

        int idx = 0;

        int token=0;
        while(idx < size && (token = fileInputStream.read()) != -1) {
            bytes[idx++] = (byte)token;
        }

        if(token == -1) {
            throw new RuntimeException("Unexpected EOF");
        }

        int value = 0;

        for(idx=0; idx < size; idx++) {
            value |= (((bytes[idx] & 0xFF) << (size - idx - 1) * 8));
        }

        return value;
    }

    /**
     * Function reads an unsigned Big Endian integer of 4 bytes.
     */
    private long readLong(FileInputStream fileInputStream) throws Exception {

        byte [] bytes = new byte[4];

        int idx = 0;

        int token = 0;
        while(idx < 4 && (token = fileInputStream.read()) != -1) {
            bytes[idx++] = (byte)token;
        }

        if(token == -1) {
            throw new RuntimeException("Unexpected EOF");
        }

        long value = 0;

        for(idx=0; idx < 4; idx++) {
            value |= (((bytes[idx] & 0xFF) << (4 - idx - 1) * 8));
        }

        return value;
    }

    public void setOutputDir(String outputDir) {
        this.outputDir = outputDir;
    }
}
