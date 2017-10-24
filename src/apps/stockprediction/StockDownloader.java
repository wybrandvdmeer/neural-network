package apps.stockprediction;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class StockDownloader {

    private static final SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");

    private static final String COLUMNS_TAG = "COLUMNS=";
    private static final String INTERVAL_TAG = "INTERVAL=";

    private static final String url = "http://finance.google.com/finance/getprices?p=100d&f=d,o,h,l,c,v&q=%s&x=%s";

    private FileOutputStream out;

    public void get(String exchange, String stock) throws Exception {
        out = new FileOutputStream(String.format("%s-%s", exchange, stock));

        URLConnection urlConnection = new URL(String.format(url, stock, exchange)).openConnection();
        InputStream input = urlConnection.getInputStream();

        List<String> lines = getLines(input);

        int interval=0;
        long beginDate=0;
        boolean parsingPrices=false;

        System.out.println(String.format("Writing file %s-%s", exchange, stock));

        for(String line : lines) {
            String [] arr = line.split(",");

            if(line.startsWith(COLUMNS_TAG)) {
                write(line.replaceFirst(COLUMNS_TAG, "").replaceAll(",", "\t"));
            }

            if(line.startsWith(INTERVAL_TAG)) {
                interval = new Integer(line.replaceFirst(INTERVAL_TAG, ""));
            }

            if(parsingPrices) {
                write(getPriceLine(beginDate, interval, new Integer(arr[0]), arr));
            }

            if(line.startsWith("a15")) {
                parsingPrices = true;
                beginDate = new Long(arr[0].substring(1));
                write(getPriceLine(beginDate, interval, 0, arr));
            }
        }

        out.close();
    }

    private void write(String s) throws Exception {
        out.write((s + "\n").getBytes());
    }

    private String getPriceLine(long date, int interval, long noOfDay, String [] columns) {
        return String.format("%s\t%s\t%s\t%s\t%s\t%s",
                formatter.format(new Date(date * 1000 + noOfDay * interval * 1000)),
                columns[1],
                columns[2],
                columns[3],
                columns[4],
                columns[5]);
    }

    private List<String> getLines(InputStream inputStream) throws Exception {
        List<String> lines = new ArrayList<>();

        int c;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        while((c = inputStream.read()) != -1) {
            if(c != '\n') {
                byteArrayOutputStream.write(c);
            } else {
                lines.add(new String(byteArrayOutputStream.toByteArray()));
                byteArrayOutputStream.reset();
            }
        }
        return lines;
    }
}
