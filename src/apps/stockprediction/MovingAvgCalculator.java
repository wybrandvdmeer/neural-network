package apps.stockprediction;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MovingAvgCalculator {
   private FileOutputStream out;

    private static final int MOVING_AVG_WINDOW_SIZE=10;
    private List<List<Double>> movingAverages = new ArrayList<>();

    public MovingAvgCalculator() {
        for(int col=0; col < 5; col++) {
            movingAverages.add(col, new ArrayList<>());
        }
    }

    public void calculateMovingAvg(String exchange, String stock) throws Exception {
        out = new FileOutputStream(String.format("%s-%s-avg", exchange, stock));
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(String.format("%s-%s", exchange, stock))));

        String line;

        boolean header = true;
        while((line = br.readLine()) != null) {
            if(header) {
                write(line);
                header = false;
                continue;
            }

            String [] columns = line.split("\t");

            for(int col=1; col <= 5; col++) {
                add2MovingAvg(movingAverages.get(col - 1), new Double(columns[col]));
            }

            write(String.format("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
                    columns[0],
                    calculateMovingAvg(movingAverages.get(0)),
                    calculateMovingAvg(movingAverages.get(1)),
                    calculateMovingAvg(movingAverages.get(2)),
                    calculateMovingAvg(movingAverages.get(3)),
                    calculateMovingAvg(movingAverages.get(4))));
        }
    }

    private void write(String s) throws Exception {
        out.write((s + "\n").getBytes());
    }

    private double calculateMovingAvg(List<Double> values) {
        return values.stream().mapToDouble(d -> d).sum() / values.size();
    }

    private void add2MovingAvg(List<Double> list, double value) {
        if(list.size() >= MOVING_AVG_WINDOW_SIZE) {
            list.remove(0);
        }
        list.add(value);
    }
}
