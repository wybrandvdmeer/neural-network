package apps.stockprediction;

import java.util.List;

public interface StockDownloader {
    public List<PriceRecord> get(String exchange, String stock) throws Exception;
}
