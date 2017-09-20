package apps.meetpuntdetector.st4;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.*;

public class St4PlotMeetpuntenInTime {

    private final FileInputStream stream;
    private String outputDir=null;

    public St4PlotMeetpuntenInTime(String st4) throws Exception {
        stream = new FileInputStream(new File(st4));
    }

    public void process(int roadNumber, char direction) throws Exception {

        int beginMeetPundID=-1, endMeetPuntID=-1;
        List<MeetPunt> selectedMeetPunten = new ArrayList<>();

        long noOfMeetPunten = readHeader(stream);
        log("Found %d meetpunten.", noOfMeetPunten);

        for(int meetPuntID=0; meetPuntID < noOfMeetPunten; meetPuntID++) {
            MeetPunt meetPunt = readMeetPunt(stream);

            if(!meetPunt.isMainRoad() || meetPunt.isSuspect()) {
                continue;
            }

            if(beginMeetPundID == -1 && roadNumber == meetPunt.getRoadNumber() && direction == meetPunt.getDirection()) {
                beginMeetPundID = meetPuntID;
            }

            if(beginMeetPundID != -1 && endMeetPuntID == -1 && (roadNumber != meetPunt.getRoadNumber() || direction != meetPunt.getDirection())) {
                endMeetPuntID = meetPuntID - 1;
            }

            if(roadNumber == meetPunt.getRoadNumber() && direction == meetPunt.getDirection()) {
                selectedMeetPunten.add(meetPunt);
            }
        }

        File directory = new File(String.format("meetpunten-%d-%c", roadNumber, direction));
        directory.mkdir();

        int beginMinute=720, endMinute=1440;

        /* Read the data. Data is in order of the meetPunten, and for each Meetpunt is starts from minute 0 to minute 1439.
        */
        for(int meetPuntID=0; meetPuntID < noOfMeetPunten; meetPuntID++) {
            if(meetPuntID >= endMeetPuntID) {
                break;
            }

            FileOutputStream out=null;

            if(meetPuntID >= beginMeetPundID && meetPuntID <= endMeetPuntID) {
                File meetPuntFile = new File(directory, String.format("meetpunt-%.1f.tsv",
                        (double)selectedMeetPunten.get(meetPuntID - beginMeetPundID).getMeterPosition()/1000));
                out = new FileOutputStream(meetPuntFile);

                //Write header.
                for(int minute=beginMinute; minute < endMinute; minute++) {
                    out.write(String.format("%d:%d", minute/60, minute%60).getBytes());
                    if (minute == endMinute - 1) {
                        out.write("\n".getBytes());
                    } else {
                        out.write("\t".getBytes());
                    }
                }
            }

            for(int minute=0; minute < 1440; minute++) {
                DataRecord dataRecord = readDataRecord(stream, minute);
                if(out != null && minute >= beginMinute && minute < endMinute) {
                    out.write(String.format("%d", dataRecord.getIntensityA()).getBytes());
                    if (minute == 1439) {
                        out.write("\n".getBytes());
                    } else {
                        out.write("\t".getBytes());
                    }
                }
            }

            if(out != null) {
                out.close();
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

    public void setOutputDir(String outputDir) {
        this.outputDir = outputDir;
    }
}
