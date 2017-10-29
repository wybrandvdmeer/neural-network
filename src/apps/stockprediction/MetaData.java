package apps.stockprediction;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MetaData {
    private static final SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
    private static final String LATEST_STOCK_DATA_DATE_TAG="latestStockDataDate=";

    public Date latestStockDate;

    private final static String NAME = "meta-data";

    public static MetaData parse() throws Exception {

        MetaData metaData = new MetaData();

        File metaFile = new File(NAME);
        if(!metaFile.exists()) {
            return metaData;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(metaFile)));

        String line;
        while((line = br.readLine()) != null) {
            metaData.latestStockDate = getDate(LATEST_STOCK_DATA_DATE_TAG, line);
        }

        return metaData;
    }

    private static Date getDate(String tag, String line)throws Exception {
        if(line.startsWith(tag)) {
            return formatter.parse(line.replaceFirst(tag,""));
        }
        return null;
    }

    private void writeTag(String tag, String value, FileOutputStream out) throws Exception {
        out.write((tag + value + "\n").getBytes());
    }

    public void write() throws Exception {
        FileOutputStream out = new FileOutputStream(NAME);
        writeTag(LATEST_STOCK_DATA_DATE_TAG, formatter.format(latestStockDate), out);
        out.close();
    }
}
