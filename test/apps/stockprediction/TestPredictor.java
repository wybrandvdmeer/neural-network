package apps.stockprediction;

import org.junit.Test;

public class TestPredictor {

    @Test
    public void testStockDownloader() throws Exception {
        StockDownloader stockDownloader = new StockDownloader();
        stockDownloader.get("AMS", "PHIA");
    }

    @Test
    public void testMovingAvgCalculator() throws Exception {
        MovingAvgCalculator movingAvgCalculator = new MovingAvgCalculator();
        movingAvgCalculator.calculateMovingAvg("AMS", "PHIA");
    }
}
