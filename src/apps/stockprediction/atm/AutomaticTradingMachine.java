package apps.stockprediction.atm;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

public class AutomaticTradingMachine {
    private static final SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd-hh24-mm-ss");
    private StockDownloader stockDownloader = new StockDownloader();

    private static List<String> stocks = new ArrayList<>();
    private static final String META_FILE="meta-data";

    public AutomaticTradingMachine() {
        stocks.add("AMS/PHIA");
    }

    public void trade() throws Exception {
        for(String exchangeAndStock : stocks) {
            String exchange = exchangeAndStock.split("/")[0];
            String stock = exchangeAndStock.split("/")[1];
            MetaData metaData;
            PriceRecordDB priceRecordDBPrices = new PriceRecordDB("prices");
            PriceRecordDB priceRecordDBAVGPrices = new PriceRecordDB("avgPrices");

            File dir = new File(exchange + "-" + stock);
            if(!dir.exists()) {
                if(!dir.mkdir()) {
                    throw new RuntimeException("Could not create dir: " + dir.getAbsolutePath());
                }
            }

            metaData = MetaData.parse();

            priceRecordDBPrices.read();
            priceRecordDBAVGPrices.read();

            List<PriceRecord> priceRecords = stockDownloader.get(exchange, stock);

            priceRecords = priceRecordDBPrices.add(priceRecords);
            priceRecordDBPrices.write();

            MovingAverageCalculator movingAverageCalculator = new MovingAverageCalculator(metaData.latestStockDate);
            List<PriceRecord> avgPriceRecords = movingAverageCalculator.average(priceRecords);

            priceRecordDBAVGPrices.add(avgPriceRecords);
            priceRecordDBAVGPrices.write();

            Predictor predictor = new Predictor(exchange, stock);
            predictor.train(avgPriceRecords, readPreviousState());

            predictor.predict(avgPriceRecords.get(avgPriceRecords.size() - 1), predictor.getPreviousState());

            metaData.latestStockDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
            metaData.write();
        }
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



    private void copy(File original, File copy) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(original)));
        FileOutputStream out = new FileOutputStream(copy);
        String line;
        while((line = br.readLine()) != null) {
            out.write(line.getBytes());
        }
        br.close();
        out.close();
    }
}
