package apps.stockprediction;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class MovingAverageCalculator {

    private static final int CLOSE_POS=0;
    private static final int HIGH_POS=1;
    private static final int LOW_POS=2;
    private static final int OPEN_POS=3;
    private static final int VOLUME_POS=4;

    private final Date lastAVGDate;
    private static final int MOVING_AVG_WINDOW_SIZE=10;
    private List<List<Double>> movingAverages = new ArrayList<>();

    public MovingAverageCalculator(Date lastAVGDate) {
        this.lastAVGDate = lastAVGDate;
        for(int col=0; col < 5; col++) {
            movingAverages.add(col, new ArrayList<>());
        }
    }

    /**
     * Assumption: priceRecords are sorted on date.
     */
    public List<PriceRecord> average(List<PriceRecord> priceRecords) {
        List<PriceRecord> avgPriceRecords = new ArrayList<>();

        int posLastProcessedPrice=0;
        for(int idx=0; idx < priceRecords.size(); idx++) {
            if(priceRecords.get(idx).date.equals(lastAVGDate)) {
                posLastProcessedPrice = idx;
                break;
            }
        }

        int idx=posLastProcessedPrice - MOVING_AVG_WINDOW_SIZE > 0 ? posLastProcessedPrice - MOVING_AVG_WINDOW_SIZE : 0;
        for(; idx <= posLastProcessedPrice; idx++) {
            add2MovingAvg(movingAverages.get(CLOSE_POS), priceRecords.get(idx).close);
            add2MovingAvg(movingAverages.get(OPEN_POS), priceRecords.get(idx).open);
            add2MovingAvg(movingAverages.get(HIGH_POS), priceRecords.get(idx).high);
            add2MovingAvg(movingAverages.get(LOW_POS), priceRecords.get(idx).low);
            add2MovingAvg(movingAverages.get(VOLUME_POS), priceRecords.get(idx).volume);
        }

        for(idx = posLastProcessedPrice + 1; idx < priceRecords.size(); idx++) {
            add2MovingAvg(movingAverages.get(CLOSE_POS), priceRecords.get(idx).close);
            add2MovingAvg(movingAverages.get(OPEN_POS), priceRecords.get(idx).open);
            add2MovingAvg(movingAverages.get(HIGH_POS), priceRecords.get(idx).high);
            add2MovingAvg(movingAverages.get(LOW_POS), priceRecords.get(idx).low);
            add2MovingAvg(movingAverages.get(VOLUME_POS), priceRecords.get(idx).volume);

            PriceRecord avgPriceRecord = new PriceRecord();
            avgPriceRecord.date = priceRecords.get(idx).date;
            avgPriceRecord.close = calculateMovingAvg(movingAverages.get(CLOSE_POS));
            avgPriceRecord.open = calculateMovingAvg(movingAverages.get(OPEN_POS));
            avgPriceRecord.high = calculateMovingAvg(movingAverages.get(OPEN_POS));
            avgPriceRecord.low = calculateMovingAvg(movingAverages.get(LOW_POS));
            avgPriceRecord.volume = calculateMovingAvg(movingAverages.get(VOLUME_POS));
            avgPriceRecords.add(avgPriceRecord);
        }

        return avgPriceRecords;
    }

    private void add2MovingAvg(List<Double> list, double value) {
        if(list.size() >= MOVING_AVG_WINDOW_SIZE) {
            list.remove(0);
        }
        list.add(value);
    }

    private double calculateMovingAvg(List<Double> values) {
        return values.stream().mapToDouble(d -> d).sum() / values.size();
    }
}
