package apps.stockprediction;

import org.junit.Test;

public class TestStockDownloader {

    @Test
    public void test() throws Exception {
        StockDownloader stockDownloader = new StockDownloader();
        stockDownloader.get("AMS", "PHIA");
    }
}
