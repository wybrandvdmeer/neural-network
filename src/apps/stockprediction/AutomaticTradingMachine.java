package apps.stockprediction;

import org.joda.time.LocalDate;

import java.io.*;
import java.util.List;

public class AutomaticTradingMachine {

    private static final String HIDDEN_STATE_FILE="hiddenStateFile";

    private StockDownloader stockDownloader;

    public void setStockDownloader(StockDownloader stockDownloader) {
        this.stockDownloader = stockDownloader;
    }

    public void getStockPrices(String exchange, String stock) throws Exception {
        String dir = exchange + "-" + stock;
        MetaData metaData = getMetaData(dir);

        PriceRecordDB priceRecordDBPrices = new PriceRecordDB(dir, "prices");
        PriceRecordDB priceRecordDBAVGPrices = new PriceRecordDB(dir, "avgPrices");

        priceRecordDBPrices.read();
        priceRecordDBAVGPrices.read();

        List<PriceRecord> priceRecords = stockDownloader.get(exchange, stock);
        priceRecordDBPrices.add(priceRecords);
        priceRecordDBPrices.write();

        MovingAverageCalculator movingAverageCalculator = new MovingAverageCalculator(metaData.mostRecentDate);
        List<PriceRecord> avgPriceRecords = movingAverageCalculator.average(priceRecordDBPrices.getPriceRecords());

        priceRecordDBAVGPrices.add(avgPriceRecords);
        priceRecordDBAVGPrices.write();

        metaData.mostRecentDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
        metaData.write(dir);
    }

    public void trainAndPredict(String exchange, String stock) throws Exception {
        String dir = exchange + "-" + stock;
        MetaData metaData = getMetaData(dir);

        PriceRecordDB priceRecordDBAVGPrices = new PriceRecordDB(dir, "avgPrices");

        LocalDate minDate = metaData.mostRecentTrainedDate.minusDays(Predictor.WINDOW_SIZE);
        List<PriceRecord> avgPriceRecords = priceRecordDBAVGPrices.read(minDate);
        if(avgPriceRecords.size() < Predictor.WINDOW_SIZE + 1) {
            return;
        }

        PriceRecord mostRecentRecord = avgPriceRecords.get(avgPriceRecords.size() - 1);
        avgPriceRecords.remove(mostRecentRecord);

        Predictor predictor = new Predictor(exchange, stock);
        predictor.train(avgPriceRecords, readHiddenState());

        writeHiddenState(predictor.getHiddenStateFirstOutput());

        metaData.mostRecentTrainedDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
        metaData.write(dir);

        predictor.predict(mostRecentRecord, predictor.getHiddenStateLastOutput());
        metaData.write(dir);
    }

    private MetaData getMetaData(String dirName) throws Exception {
        File dir = new File(dirName);
        if (!dir.exists()) {
            if (!dir.mkdir()) {
                throw new RuntimeException("Could not create dir: " + dir.getAbsolutePath());
            }
        }

        if (!dir.isDirectory()) {
            throw new RuntimeException(String.format("Dir %s is not a directory.", dir.getAbsolutePath()));
        }

        return MetaData.parse(dir);
    }

    private void writeHiddenState(double [] hiddenState) throws Exception {
        File hiddenStateFile = new File(HIDDEN_STATE_FILE);
        FileOutputStream out = new FileOutputStream(hiddenStateFile);

        for(int idx=0; idx < hiddenState.length; idx++) {
            out.write(String.format("%f", hiddenState[idx]).getBytes());
            if(idx < hiddenState.length - 2) {
                out.write("\t".getBytes());
            }
        }
    }

    private double [] readHiddenState() throws Exception {
        File hiddenState = new File(HIDDEN_STATE_FILE);
        if(!hiddenState.exists()) {
            return null;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(hiddenState)));

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
