package apps.stockprediction;

import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import rnn.Network;

import java.util.List;

public class Predictor {
    private static final int INPUT_DATE=0;
    private static final int INPUT_OPEN=1;
    private static final int INPUT_CLOSE=2;
    private static final int INPUT_VOLUME=3;

    private static final int HIGHER_1_PERCENT_POS=0;
    private static final int BETWEEN_0_AND_1_PERCENT_POS=1;
    private static final int BETWEEN_0_AND_1_PERCENT_NEG_POS=2;
    private static final int HIGHER_1_PERCENT_NEG_POS=3;

    public static final int WINDOW_SIZE=10;

    private static final DateTimeFormatter yyyymmdd = DateTimeFormat.forPattern("yyyy-MM-dd");
    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("hh:mm:ss");

    private static final double ERROR_LIMIT = 0.0001;

    private Network network;

    public Predictor(String exchange, String stock) throws Exception {
        network = new Network(exchange + "-" + stock + "-network", new int[] {4, 30, 10, 4}, WINDOW_SIZE, true);
        network.setWeightFileDir(exchange + "-" + stock);
        network.setLearningRate(0.1);
        network.read();

        // We are only interested in the last output.
        network.setBeginErrorOutput(WINDOW_SIZE - 1);
    }

    public Prediction predict(List<PriceRecord> priceRecords) {
        System.out.println("Last date prediction set: " + yyyymmdd.print(priceRecords.get(priceRecords.size() - 1).date));

        double [][] inputs = new double[WINDOW_SIZE][4];
        for(int idx=0; idx < priceRecords.size(); idx++) {
            inputs[idx][INPUT_DATE] = scaleDate(priceRecords.get(idx).date);
            inputs[idx][INPUT_OPEN] = scalePrice(priceRecords.get(idx).open);
            inputs[idx][INPUT_CLOSE] = scalePrice(priceRecords.get(idx).close);
            inputs[idx][INPUT_VOLUME] = scaleVolume(priceRecords.get(idx).volume);
        }

        network.passForward(inputs);

        int highestRow=-1;
        double highestPercentage=0;
        for(int row=0; row < network.getOutputVector().getRowDimension(); row++) {
            if(highestPercentage < network.getOutputVector().get(row, 0)) {
                highestRow = row;
                highestPercentage = network.getOutputVector().get(row, 0);
            }
        }

        System.out.println("Percentages:");
        System.out.println("close > 1: " + network.getOutputVector().get(HIGHER_1_PERCENT_POS,0));
        System.out.println("0 < close < 1 %:  " + network.getOutputVector().get(BETWEEN_0_AND_1_PERCENT_POS,0));
        System.out.println("-1 < close < 1 %: " + network.getOutputVector().get(BETWEEN_0_AND_1_PERCENT_NEG_POS,0));
        System.out.println("close < -2: " + network.getOutputVector().get(HIGHER_1_PERCENT_NEG_POS,0));

        return Prediction.values()[highestRow];
    }

    public void train(List<PriceRecord> priceRecords) throws Exception {
        if(priceRecords.size() < WINDOW_SIZE) {
            throw new RuntimeException("Window rnn is bigger than training batch.");
        }

        double [][] inputs = new double[WINDOW_SIZE][4];
        double [][] targets = new double[WINDOW_SIZE][4];

        int idx1=0;

        while(idx1 + WINDOW_SIZE <= priceRecords.size()) {
            System.out.println("Train window: " + yyyymmdd.print(priceRecords.get(idx1).date));
            for(int idx2=0; idx2 < WINDOW_SIZE; idx2++) {
                PriceRecord priceRecord = priceRecords.get(idx1 + idx2);

                inputs[idx2][INPUT_DATE] = scaleDate(priceRecord.date);
                inputs[idx2][INPUT_OPEN] = scalePrice(priceRecord.open);
                inputs[idx2][INPUT_CLOSE] = scalePrice(priceRecord.close);
                inputs[idx2][INPUT_VOLUME] = scaleVolume(priceRecord.volume);
            }

            initArray(targets);

            for(int idx2=1; idx2 < WINDOW_SIZE; idx2++) {

                double close = priceRecords.get(idx1 + idx2).close;
                double previousClose = priceRecords.get(idx1 + idx2 - 1).close;
                double delta = (close / previousClose - 1) * 100;

                if(delta > 1) {
                    targets[idx2][HIGHER_1_PERCENT_POS] = 1;
                } else if(delta > 0) {
                    targets[idx2][BETWEEN_0_AND_1_PERCENT_POS] = 1;
                } else if(delta > -1) {
                    targets[idx2][BETWEEN_0_AND_1_PERCENT_NEG_POS] = 1;
                } else {
                    targets[idx2][HIGHER_1_PERCENT_NEG_POS] = 1;
                }
            }

            System.out.println("Begin training: " + formatter.print(new LocalDateTime()));
            int iterations = network.learn(inputs, targets, ERROR_LIMIT, 0);
            System.out.println(String.format("End training: %s, iterations: %d.", formatter.print(new LocalDateTime()), iterations));

            idx1++;
        }

        network.write();
    }

    private void initArray(double [][] array) {
        for(int idx1=0; idx1 < array.length; idx1++) {
            for(int idx2=0; idx2 < array[idx1].length; idx2++) {
                array[idx1][idx2] = 0;
            }
        }
    }

    private double scaleDate(LocalDate date) {
        return (double)date.getDayOfWeek() / 5 - 0.01;
    }

    private double scalePrice(double price) {
        return price / 100;
    }

    private double scaleVolume(double volume) {
        return volume / 5000000;
    }
}
