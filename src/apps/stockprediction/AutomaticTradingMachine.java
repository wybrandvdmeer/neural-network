package apps.stockprediction;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

public class AutomaticTradingMachine {
    private static final SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd-hh24-mm-ss");
    private StockDownloader stockDownloader;

    private static List<String> stocks = new ArrayList<>();
    private static final String META_FILE="meta-data";

    public void setStockDownloader(StockDownloader stockDownloader) {
        this.stockDownloader = stockDownloader;
    }

    public void getStockPrices(String exchange, String stock) throws Exception {
        MetaData metaData;

        File dir = new File(exchange + "-" + stock);
        if (!dir.exists()) {
            if (!dir.mkdir()) {
                throw new RuntimeException("Could not create dir: " + dir.getAbsolutePath());
            }
        }

        if (!dir.isDirectory()) {
            throw new RuntimeException(String.format("Dir %s is not a directory.", dir.getAbsolutePath()));
        }

        metaData = MetaData.parse(dir);

        PriceRecordDB priceRecordDBPrices = new PriceRecordDB(dir.getName(), "prices");
        PriceRecordDB priceRecordDBAVGPrices = new PriceRecordDB(dir.getName(), "avgPrices");

        priceRecordDBPrices.read();
        priceRecordDBAVGPrices.read();

        List<PriceRecord> priceRecords = stockDownloader.get(exchange, stock);
        priceRecordDBPrices.add(priceRecords);
        priceRecordDBPrices.write();

        MovingAverageCalculator movingAverageCalculator = new MovingAverageCalculator(metaData.latestStockDate);
        List<PriceRecord> avgPriceRecords = movingAverageCalculator.average(priceRecordDBPrices.getPriceRecords());

        priceRecordDBAVGPrices.add(avgPriceRecords);
        priceRecordDBAVGPrices.write();

        metaData.latestStockDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
        metaData.write(dir);

    }

    public void trade() throws Exception {
        /*
        for(String exchangeAndStock : stocks) {

            Predictor predictor = new Predictor(exchange, stock);
            predictor.train(avgPriceRecords, readPreviousState());

            predictor.predict(avgPriceRecords.get(avgPriceRecords.size() - 1), predictor.getPreviousState());

            metaData.latestStockDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
            metaData.write();
        }
        */
    }

    private double [] readPreviousState() throws Exception {

        File previousState = new File("previousState");
        if(!previousState.exists()) {
            return null;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(previousState)));

        String line;
        if((line = br.readLine()) == null) {
            throw new RuntimeException("Corrupt previous state.");
        }

        String [] columns = line.split("\t");

        double [] previousStateArr = new double[columns.length];

        for(int idx=0; idx < columns.length; idx++) {
            previousStateArr[idx] = Double.parseDouble(columns[idx]);
        }

        return previousStateArr;
    }
}
