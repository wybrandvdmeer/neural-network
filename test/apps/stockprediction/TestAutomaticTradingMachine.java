package apps.stockprediction;

import org.joda.time.LocalDate;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.junit.After;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestAutomaticTradingMachine {

    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("yyyy-MM-dd");

    @Test
    public void testDownloadFirstTimePrices() throws Exception {
        PriceRecord p0 = new PriceRecord();
        p0.date = formatter.parseLocalDate("2017-01-01");
        p0.open = 10;
        p0.close = 15;
        p0.low = 8;
        p0.high = 16;
        p0.volume = 2000;

        PriceRecord p1 = new PriceRecord();
        p1.date = formatter.parseLocalDate("2017-01-02");
        p1.open = 10;
        p1.close = 15;
        p1.low = 8;
        p1.high = 16;
        p1.volume = 2000;

        PriceRecord p2 = new PriceRecord();
        p2.date = formatter.parseLocalDate("2017-01-03");
        p2.open = 10;
        p2.close = 15;
        p2.low = 8;
        p2.high = 16;
        p2.volume = 2000;

        List<PriceRecord> priceRecords = new ArrayList<>();
        priceRecords.add(p0);
        priceRecords.add(p1);
        priceRecords.add(p2);

        AutomaticTradingMachine atm = new AutomaticTradingMachine();
        atm.setStockDownloader((exchange, stock) -> {
            return priceRecords;
        });

        atm.getStockPrices("AMS", "PHIA");

        File dir = new File("AMS-PHIA");
        assertTrue(dir.exists() && dir.isDirectory());

        File pricesFile = new File(dir, "prices");
        File avgPricesFile = new File(dir, "avgPrices");

        assertTrue(pricesFile.exists());
        assertTrue(avgPricesFile.exists());

        PriceRecordDB pricesDB = new PriceRecordDB("AMS-PHIA", "prices");
        PriceRecordDB avgPricesDB = new PriceRecordDB("AMS-PHIA", "avgPrices");

        List<PriceRecord> prices = pricesDB.read();
        List<PriceRecord> avgPrices = avgPricesDB.read();

        equalPrices(prices, priceRecords);
        equalPrices(avgPrices, priceRecords);

        MetaData metaData = MetaData.parse(dir);
        assertEquals(formatter.parseLocalDate("2017-01-03"), metaData.mostRecentDate);
    }

    @Test
    public void testDownloadSecondTimePrices() throws Exception {
        List<PriceRecord> prices = new ArrayList<>();

        LocalDate firstDate = formatter.parseLocalDate("2017-01-01");
        for(int idx=0; idx < 11; idx++) {
            PriceRecord priceRecord = new PriceRecord();
            priceRecord.date = firstDate.plusDays(idx);
            priceRecord.open = 10;
            priceRecord.close = 15;
            priceRecord.low = 8;
            priceRecord.high = 16;
            priceRecord.volume = 2000;
            prices.add(priceRecord);
        }

        File dir = new File("AMS-PHIA");
        dir.mkdir();

        PriceRecordDB pricesDB = new PriceRecordDB("AMS-PHIA", "prices");
        PriceRecordDB avgPricesDB = new PriceRecordDB("AMS-PHIA", "avgPrices");

        MovingAverageCalculator movingAverageCalculator = new MovingAverageCalculator(null);
        List<PriceRecord> avgPrices = movingAverageCalculator.average(prices);

        pricesDB.add(prices);
        avgPricesDB.add(avgPrices);

        pricesDB.write();
        avgPricesDB.write();

        MetaData metaData = MetaData.parse(dir);
        metaData.mostRecentDate = avgPrices.get(avgPrices.size() - 1).date;
        metaData.write(dir.getName());

        /* New download: overlaps existing data. */
        PriceRecord p0 = new PriceRecord();
        p0.date = formatter.parseLocalDate("2017-01-10");
        p0.open = 10;
        p0.close = 15;
        p0.low = 8;
        p0.high = 16;
        p0.volume = 2000;

        PriceRecord p1 = new PriceRecord();
        p1.date = formatter.parseLocalDate("2017-01-11");
        p1.open = 10;
        p1.close = 15;
        p1.low = 8;
        p1.high = 16;
        p1.volume = 2000;

        PriceRecord p2 = new PriceRecord();
        p2.date = formatter.parseLocalDate("2017-01-12");
        p2.open = 20;
        p2.close = 30;
        p2.low = 16;
        p2.high = 32;
        p2.volume = 4000;

        PriceRecord p3 = new PriceRecord();
        p3.date = formatter.parseLocalDate("2017-01-13");
        p3.open = 20;
        p3.close = 30;
        p3.low = 16;
        p3.high = 32;
        p3.volume = 4000;

        List<PriceRecord> priceRecords = new ArrayList<>();
        priceRecords.add(p0);
        priceRecords.add(p1);
        priceRecords.add(p2);
        priceRecords.add(p3);

        AutomaticTradingMachine atm = new AutomaticTradingMachine();
        atm.setStockDownloader((exchange, stock) -> {
            return priceRecords;
        });

        atm.getStockPrices("AMS", "PHIA");

        pricesDB = new PriceRecordDB("AMS-PHIA", "prices");
        avgPricesDB = new PriceRecordDB("AMS-PHIA", "avgPrices");

        List<PriceRecord> pricesInDB = pricesDB.read();
        avgPrices = avgPricesDB.read();

        metaData = MetaData.parse(dir);
        assertEquals(formatter.parseLocalDate("2017-01-13"), metaData.mostRecentDate);

        assertEquals(11, avgPrices.get(avgPrices.size() - 2).open, 0.0001);
        assertEquals(12, avgPrices.get(avgPrices.size() - 1).open, 0.0001);

        assertEquals(2200, avgPrices.get(avgPrices.size() - 2).volume, 0.0001);
        assertEquals(2400, avgPrices.get(avgPrices.size() - 1).volume, 0.0001);

        prices.add(p2);
        prices.add(p3);

        equalPrices(prices, pricesInDB);
    }

    private void equalPrices(List<PriceRecord> prices1, List<PriceRecord> prices2) {
        assertTrue(prices1.size() == prices2.size());

        Collections.sort(prices1, (d1, d2) -> d1.date.compareTo(d2.date));
        Collections.sort(prices2, (d1, d2) -> d1.date.compareTo(d2.date));

        for(int idx=0; idx < prices1.size(); idx++) {
            equalPrice(prices1.get(idx), prices2.get(idx));
        }
    }

    private void equalPrice(PriceRecord price1, PriceRecord price2) {
        assertTrue(price1.date.equals(price2.date));
        assertTrue(price1.close == price2.close);
        assertTrue(price1.open == price2.open);
        assertTrue(price1.high == price2.high);
        assertTrue(price1.low == price2.low);
        assertTrue(price1.volume == price2.volume);
    }

    @After
    public void after() {
        delete(new File("AMS-PHIA"));
    }

    private void delete(File file) {
        if(!file.exists()) {
            return;
        }
        if(file.isDirectory()) {
            for(File subFile : file.listFiles()) {
                delete(subFile);
            }
        }
        file.delete();
    }
}
