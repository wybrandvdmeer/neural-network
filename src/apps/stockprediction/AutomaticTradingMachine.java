package apps.stockprediction;

import org.joda.time.DateTimeConstants;
import org.joda.time.LocalDate;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.*;
import java.util.List;

public class AutomaticTradingMachine {

    public static final String HIDDEN_STATE_FILE="hiddenStateFile";
    public static final String PREDICTION_FILE="predictions";

    private StockDownloader stockDownloader;

    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("yyyy-MM-dd");

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

        LocalDate minDate = metaData.mostRecentTrainedDate != null ? metaData.mostRecentTrainedDate.minusDays(Predictor.WINDOW_SIZE) : null;
        List<PriceRecord> avgPriceRecords = priceRecordDBAVGPrices.read(minDate);
        if(avgPriceRecords.size() < Predictor.WINDOW_SIZE + 1) {
            return;
        }

        PriceRecord mostRecentRecord = avgPriceRecords.get(avgPriceRecords.size() - 1);
        avgPriceRecords.remove(mostRecentRecord);

        Predictor predictor = new Predictor(exchange, stock);
        predictor.train(avgPriceRecords, readHiddenState(dir));

        writeHiddenState(dir, predictor.getHiddenStateFirstOutput());

        metaData.mostRecentTrainedDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
        metaData.write(dir);

        avgPriceRecords.remove(avgPriceRecords.get(0));
        avgPriceRecords.add(mostRecentRecord);

        Prediction prediction = predictor.predict(avgPriceRecords, predictor.getHiddenStateLastOutput());
        writePrediction(dir, prediction, mostRecentRecord);
    }

    private LocalDate nextWorkingDay(LocalDate localDate) {
        LocalDate nextWorkingDay = localDate;
        while(true) {
            nextWorkingDay = nextWorkingDay.plusDays(1);
            if(nextWorkingDay.getDayOfWeek() >= DateTimeConstants.MONDAY && nextWorkingDay.getDayOfWeek() <= DateTimeConstants.FRIDAY) {
                return nextWorkingDay;
            }
        }
    }

    private void writePrediction(String dir, Prediction prediction, PriceRecord priceRecord) throws Exception {
        LocalDate predictionDate = nextWorkingDay(priceRecord.date);
        BufferedWriter bw = new BufferedWriter(new FileWriter(dir + "/" + PREDICTION_FILE, true));

        double lower=0, upper=0;
        switch (prediction) {
            case BETWEEN_0_AND_1_PERCENT:
                lower = priceRecord.close;
                upper = priceRecord.close * 1.01;
                break;

            case BETWEEN_1_AND_2_PERCENT:
                lower = priceRecord.close * 1.01;
                upper = priceRecord.close * 1.02;
                break;

            case HIGHER_2_PERCENT:
                upper = priceRecord.close * 1.02;
                lower = 0;
                break;

            case BETWEEN_0_AND_1_PERCENT_NEG:
                upper = priceRecord.close;
                lower = priceRecord.close * 0.99;
                break;

            case BETWEEN_1_AND_2_PERCENT_NEG:
                upper = priceRecord.close * 0.99;
                lower = priceRecord.close * 0.98;
                break;

            case HIGHER_2_PERCENT_NEG:
                upper = 0;
                lower = priceRecord.close * 0.98;
                break;
        }

        bw.write(String.format("%s: %s %.3f %.3f\n", formatter.print(predictionDate),
                                        prediction,
                                        lower,
                                        upper));
        bw.close();
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

    private void writeHiddenState(String dir, double [] hiddenState) throws Exception {
        File hiddenStateFile = new File(dir, HIDDEN_STATE_FILE);
        FileOutputStream out = new FileOutputStream(hiddenStateFile);

        for(int idx=0; idx < hiddenState.length; idx++) {
            out.write(String.format("%f", hiddenState[idx]).getBytes());
            if(idx < hiddenState.length) {
                out.write("\t".getBytes());
            }
        }
    }

    private double [] readHiddenState(String dir) throws Exception {
        File hiddenState = new File(dir, HIDDEN_STATE_FILE);
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
