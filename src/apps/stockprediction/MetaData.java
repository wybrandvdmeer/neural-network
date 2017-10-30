package apps.stockprediction;

import org.joda.time.LocalDate;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.*;

public class MetaData {
    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("yyyy-MM-dd");
    private static final String MOST_RECENT_DATE_TAG ="mostRecentDate=";
    private static final String MOST_RECENT_TRAINED_DATE_TAG="mostRecentTrainedDate";

    public LocalDate mostRecentDate;
    public LocalDate mostRecentTrainedDate;

    private final static String NAME = "meta-data";

    public static MetaData parse(File dir) throws Exception {

        MetaData metaData = new MetaData();

        File metaFile = new File(dir, NAME);
        if(!metaFile.exists()) {
            return metaData;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(metaFile)));

        String line;
        while((line = br.readLine()) != null) {
            metaData.mostRecentDate = getDate(MOST_RECENT_DATE_TAG, line);
            metaData.mostRecentTrainedDate = getDate(MOST_RECENT_TRAINED_DATE_TAG, line);
        }

        return metaData;
    }

    private static LocalDate getDate(String tag, String line)throws Exception {
        if(line.startsWith(tag)) {
            return formatter.parseLocalDate(line.replaceFirst(tag,""));
        }
        return null;
    }

    private void writeTag(String tag, String value, FileOutputStream out) throws Exception {
        out.write((tag + value + "\n").getBytes());
    }

    public void write(String dir) throws Exception {
        FileOutputStream out = new FileOutputStream(new File(dir, NAME));
        if(mostRecentDate != null) {
            writeTag(MOST_RECENT_DATE_TAG, formatter.print(mostRecentDate), out);
        }
        if(mostRecentTrainedDate != null) {
            writeTag(MOST_RECENT_TRAINED_DATE_TAG, formatter.print(mostRecentTrainedDate), out);
        }
        out.close();
    }
}
