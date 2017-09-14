package apps.meetpuntdetector.st4;

import java.io.File;
import java.io.FileInputStream;

public class St4Reader {

    public St4Reader(File st4) throws Exception {

        FileInputStream fileInputStream = new FileInputStream(st4);
        long noOfMeasurements = readHeader(fileInputStream);

        log("Found %d meetpunten.", noOfMeasurements);

        for(int idx=0; idx < noOfMeasurements; idx++) {
            MeetPunt meetPunt = readMeetPunt(fileInputStream);
            log("Found meetpunt %s.", meetPunt);
        }

        /* Read the data. Data is in order of the meetPunten, and for each Meetpunt is starts from minute 0 to minute 1439.
        */
        for(int meetPuntIdx=0; meetPuntIdx < noOfMeasurements; meetPuntIdx++) {
            for(int minute=0; minute < 1440; minute++) {
                DataRecord dataRecord = readDataRecord(fileInputStream, minute);
                log("Found datarecord %s.", dataRecord);
            }
        }
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
