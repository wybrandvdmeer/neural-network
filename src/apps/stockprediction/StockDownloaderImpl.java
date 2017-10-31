package apps.stockprediction;

import org.joda.time.LocalDate;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;

public class StockDownloaderImpl implements StockDownloader {

    private static final String INTERVAL_TAG = "INTERVAL=";

    private static final String url = "http://finance.google.com/finance/getprices?p=100d&f=d,o,h,l,c,v&q=%s&x=%s";

    public List<PriceRecord> get(String exchange, String stock) throws Exception {

        List<PriceRecord> priceRecords = new ArrayList<>();

        URLConnection urlConnection = new URL(String.format(url, stock, exchange)).openConnection();
        InputStream input = urlConnection.getInputStream();

        List<String> lines = getLines(input);

        int interval=0;
        long beginDate=0;
        boolean parsingPrices=false;

        System.out.println(String.format("Writing file %s-%s", exchange, stock));

        for(String line : lines) {
            String [] arr = line.split(",");

            if(line.startsWith("TIMEZONE_OFFSET")) {
                parsingPrices = false;
            } else if(line.startsWith(INTERVAL_TAG)) {
                interval = new Integer(line.replaceFirst(INTERVAL_TAG, ""));
            } else if(parsingPrices) {
                priceRecords.add(getPriceRecord(beginDate, interval, new Integer(arr[0]), arr));
            } else if(line.startsWith("a15")) {
                parsingPrices = true;
                beginDate = new Long(arr[0].substring(1));
                priceRecords.add(getPriceRecord(beginDate, interval, 0, arr));
            }
        }

        return priceRecords;
    }

    private PriceRecord getPriceRecord(long date, int interval, long noOfDay, String [] columns) {
        return new PriceRecord(
                new LocalDate(date * 1000 + noOfDay * interval * 1000),
                new Double(columns[1]),
                new Double(columns[2]),
                new Double(columns[3]),
                new Double(columns[4]),
                new Double(columns[5]));
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
