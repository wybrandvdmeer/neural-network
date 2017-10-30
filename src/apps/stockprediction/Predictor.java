package apps.stockprediction;

import org.joda.time.LocalDate;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import rnn.Network;

import java.util.List;

public class Predictor {
    private static final int INPUT_DATE=0;
    private static final int INPUT_OPEN=1;
    private static final int INPUT_CLOSE=2;
    private static final int INPUT_VOLUME=3;

    private static final int HIGHER_05_PERCENT=0;
    private static final int BETWEEN_0_AND_05_PERCENT=1;
    private static final int BETWEEN_0_AND_05_PERCENT_NEG=2;
    private static final int HIGHER_05_PERCENT_NEG=3;

    public static final int WINDOW_SIZE=5;

    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("hh:mm:ss");

    private static final double ERROR_LIMIT = 0.0001;

    private Network network;

    public Predictor(String exchange, String stock) throws Exception {
        network = new Network(exchange + "-" + stock + "-network", new int[] {4, 10, 10, 4}, WINDOW_SIZE);
        network.setLearningRate(0.1);
        network.read();

        // Ignore the first output (since it has no previous input).
        network.setBeginErrorOutput(1);
    }

    public void predict(PriceRecord priceRecord, double [] previousState) {
        double [] input = new double[4];
        input[INPUT_DATE] = scaleDate(priceRecord.date);
        input[INPUT_OPEN] = scalePrice(priceRecord.open);
        input[INPUT_CLOSE] = scalePrice(priceRecord.close);
        input[INPUT_VOLUME] = scaleVolume(priceRecord.volume);

        network.passForward(input, previousState);
    }

    public void train(List<PriceRecord> priceRecords, double [] previousState) throws Exception {
        double [][] inputs = new double[WINDOW_SIZE][4];
        double [][] targets = new double[WINDOW_SIZE][4];

        int idx1=0;

        while(idx1 + WINDOW_SIZE <= priceRecords.size()) {
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

                if(delta > 0.5) {
                    targets[idx2][HIGHER_05_PERCENT] = 0.99;
                } else if(delta >= 0) {
                    targets[idx2][BETWEEN_0_AND_05_PERCENT] = 0.99;
                } else if(delta >= -0.5) {
                    targets[idx2][BETWEEN_0_AND_05_PERCENT_NEG] = 0.99;
                } else {
                    targets[idx2][HIGHER_05_PERCENT_NEG] = 0.99;
                }
            }

            System.out.println("Begin training: " + formatter.print(new LocalDate()));

            int iterations = network.learn(inputs, targets, previousState, ERROR_LIMIT, 0);

            System.out.println(String.format("End training: %s, iterations: %d.", formatter.print(new LocalDate()), iterations));

            idx1++;
        }

        network.write();
    }

    public double [] getHiddenStateLastOutput() {
        return network.getHiddenState(WINDOW_SIZE - 1).getArray()[0];
    }

    public double [] getHiddenStateFirstOutput() {
        return network.getHiddenState(0).getArray()[0];
    }

    private void initArray(double [][] array) {
        for(int idx1=0; idx1 < array.length; idx1++) {
            for(int idx2=0; idx2 < array[idx1].length; idx2++) {
                array[idx1][idx2] = 0.01;
            }
        }
    }

    private double scaleDate(LocalDate date) {
        return date.getDayOfWeek() / 5 - 0.01;
    }

    private double scalePrice(double price) {
        return price / 100;
    }

    private double scaleVolume(double volume) {
        return volume / 5000000;
    }
}
