package apps.stockprediction;

import org.joda.time.DateTimeConstants;
import org.joda.time.LocalDate;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.*;
import java.util.List;
import java.util.Locale;

public class AutomaticTradingMachine {

    public static final String HIDDEN_STATE_FILE="hiddenStateFile";
    public static final String PREDICTION_FILE="predictions";

    private StockDownloader stockDownloader;

    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("yyyy-MM-dd");

    public AutomaticTradingMachine() {
        setStockDownloader(new StockDownloaderImpl());
    }

    void setStockDownloader(StockDownloader stockDownloader) {
        this.stockDownloader = stockDownloader;
    }

    void getStockPrices(String exchange, String stock) throws Exception {
        String dir = exchange + "-" + stock;
        MetaData metaData = getMetaData(dir);

        PriceRecordDB priceRecordDBPrices = new PriceRecordDB(dir, "prices");
        PriceRecordDB priceRecordDBAVGPrices = new PriceRecordDB(dir, "avgPrices");

        List<PriceRecord> priceRecords = stockDownloader.get(exchange, stock);
        priceRecordDBPrices.add(priceRecords);
        priceRecordDBPrices.write();

        MovingAverageCalculator movingAverageCalculator = new MovingAverageCalculator(metaData.mostRecentDate);
        List<PriceRecord> avgPriceRecords = movingAverageCalculator.average(priceRecordDBPrices.getPriceRecords());

        if(avgPriceRecords.size() > 0) {
            priceRecordDBAVGPrices.add(avgPriceRecords);
            priceRecordDBAVGPrices.write();

            metaData.mostRecentDate = avgPriceRecords.get(avgPriceRecords.size() - 1).date;
            metaData.write(dir);
        }
    }

    void trainAndPredict(String exchange, String stock) throws Exception {
        String dir = exchange + "-" + stock;
        MetaData metaData = getMetaData(dir);

        PriceRecordDB priceRecordDBAVGPrices = new PriceRecordDB(dir, "avgPrices");
        List<PriceRecord> avgPriceRecords = priceRecordDBAVGPrices.get();

        /* We always train until (inclusive) the most recent record minus 1.
        */
        if(avgPriceRecords.get(avgPriceRecords.size() - 2).date.equals(metaData.mostRecentTrainedDate)) {
            return;
        }

        LocalDate minDate = determineFirstDateTrainingBatch(avgPriceRecords, metaData.mostRecentTrainedDate);
        if(minDate == null) {
            return;
        }

        List<PriceRecord> trainingInput = priceRecordDBAVGPrices.get(minDate);

        PriceRecord mostRecentRecord = trainingInput.get(trainingInput.size() - 1);
        trainingInput.remove(mostRecentRecord);

        Predictor predictor = new Predictor(exchange, stock);
        predictor.train(trainingInput);

        metaData.mostRecentTrainedDate = trainingInput.get(trainingInput.size() - 1).date;
        metaData.write(dir);

        List<PriceRecord> predictionSet = priceRecordDBAVGPrices.get(Predictor.WINDOW_SIZE);

        Prediction prediction = predictor.predict(predictionSet);
        writePrediction(dir, prediction, predictionSet.get(predictionSet.size() - 1));
    }

    private LocalDate determineFirstDateTrainingBatch(List<PriceRecord> priceRecords, LocalDate mostRecentTrainedDate) {
        if(mostRecentTrainedDate == null) {
            return priceRecords.get(0).date;
        }

        for(int idx=0; idx < priceRecords.size(); idx++) {
            if(priceRecords.get(idx).date.equals(mostRecentTrainedDate)) {
                return idx > Predictor.WINDOW_SIZE ? priceRecords.get(idx - Predictor.WINDOW_SIZE).date : null;
            }
        }
        return null;
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

            case HIGHER_1_PERCENT:
                upper = priceRecord.close * 1.01;
                lower = 0;
                break;

            case BETWEEN_0_AND_1_PERCENT_NEG:
                upper = priceRecord.close;
                lower = priceRecord.close * 0.99;
                break;

            case HIGHER_1_PERCENT_NEG:
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

    public static void main(String [] args) throws Exception {
        Locale.setDefault(Locale.US);
        AutomaticTradingMachine atm = new AutomaticTradingMachine();
        atm.getStockPrices(args[0], args[1]);
        atm.trainAndPredict(args[0], args[1]);
    }
}
