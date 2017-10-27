package apps.stockprediction.atm;

import rnn.Network;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
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

    private static final int WINDOW_SIZE=5;

    private static final SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
    private static final SimpleDateFormat timeFormatter = new SimpleDateFormat("hh:mm:ss");

    private final String exchange, stock;

    private static final double ERROR_LIMIT = 0.0001;

    private Network network;

    public Predictor(String exchange, String stock) throws Exception {
        this.exchange = exchange;
        this.stock = stock;

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

        while(idx1 + WINDOW_SIZE < priceRecords.size()) {
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

            System.out.println("Begin training: " + timeFormatter.format(new Date()));

            int iterations = network.learn(inputs, targets, previousState, ERROR_LIMIT, 0);

            System.out.println(String.format("End training: %s, iterations: %d.", timeFormatter.format(new Date()), iterations));

            idx1++;
        }

        network.write();
    }

    public double [] getPreviousState() {
        return network.getPreviousState();
    }

    private void initArray(double [][] array) {
        for(int idx1=0; idx1 < array.length; idx1++) {
            for(int idx2=0; idx2 < array[idx1].length; idx2++) {
                array[idx1][idx2] = 0.01;
            }
        }
    }

    private double scaleDate(Date date) {
        Calendar c = Calendar.getInstance();
        c.setTime(date);
        return (c.get(Calendar.DAY_OF_WEEK) - Calendar.MONDAY) / 5 + 0.01;
    }

    private double scalePrice(double price) {
        return price / 100;
    }

    private double scaleVolume(double volume) {
        return volume / 5000000;
    }
}
