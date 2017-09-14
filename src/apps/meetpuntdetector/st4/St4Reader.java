package apps.meetpuntdetector.st4;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class St4Reader {

    private final FileInputStream stream;

    public St4Reader(File st4) throws Exception {
        stream = new FileInputStream(st4);
    }

    public void process(List<Sample> samples) throws Exception {

        Map<Integer, Sample> meetPuntIdx2Sample = new HashMap<>();

        long noOfMeetPunten = readHeader(stream);

        log("Found %d meetpunten.", noOfMeetPunten);

        for(int meetPuntSequence=0; meetPuntSequence < noOfMeetPunten; meetPuntSequence++) {
            MeetPunt meetPunt = readMeetPunt(stream);

            if(!meetPunt.isMainRoad() || meetPunt.isSuspect()) {
                continue;
            }

            for(Iterator<Sample> it = samples.iterator(); it.hasNext();) {
                Sample sample = it.next();
                if(sample.getRoadNumber() == meetPunt.getRoadNumber() && sample.getDirection() == meetPunt.getDirection()) {
                    log("Found Sampled meetpunt %s.", meetPunt);
                    meetPuntIdx2Sample.put(meetPuntSequence, sample);
                    it.remove();
                }
            }

            if(samples.isEmpty()) {
                break;
            }
        }

        Sample sample=null;
        int meetPuntSequencePerRoad=0;

        int intensityA=0, intensityB=0, velocityA=0, velocityB=0;

        /* Read the data. Data is in order of the meetPunten, and for each Meetpunt is starts from minute 0 to minute 1439.
        */
        for(int meetPuntSequence=0; meetPuntSequence < noOfMeetPunten; meetPuntSequence++) {
            if(sample == null) {
                for (int key : meetPuntIdx2Sample.keySet()) {
                    if (key == meetPuntSequence) {
                        break;
                    }
                }
                sample = meetPuntIdx2Sample.get(meetPuntSequence);
                meetPuntSequencePerRoad = 0;
            }

            intensityA=intensityB=velocityA=velocityB=0;

            for(int minute=0; minute < 1440; minute++) {
                DataRecord dataRecord = readDataRecord(stream, minute);

                if(sample != null && sample.meetPuntInRange(meetPuntSequencePerRoad) && sample.minuteInRange(minute)) {
                    intensityA += dataRecord.getIntensityA();
                    intensityB += dataRecord.getIntensityB();
                    velocityA += dataRecord.getVelocityA();
                    velocityB += dataRecord.getVelocityB();
                }
            }

            if(sample != null && sample.meetPuntInRange(meetPuntSequencePerRoad)) {
                writeSample(meetPuntSequence, intensityA, intensityB, velocityA, velocityB);
            }

            if(sample != null && sample.meetPuntAboveRange(meetPuntSequencePerRoad)) {
                meetPuntIdx2Sample.values().remove(sample);
                sample = null;
            }

            if(meetPuntIdx2Sample.isEmpty()) {
                break;
            }

            meetPuntSequencePerRoad++;
        }
    }

    private void writeSample(int meetPuntID, int intensityA, int intentityB, int velocityA, int velocityB) throws Exception {
        FileOutputStream out = new FileOutputStream(new File(String.format("%d", meetPuntID)));
        out.write(String.format("%d %d %d %d", intensityA, intentityB, velocityA, velocityB).getBytes());
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
}
