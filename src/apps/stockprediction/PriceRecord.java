package apps.stockprediction;

import org.joda.time.LocalDate;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.text.SimpleDateFormat;

public class PriceRecord {
    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("yyyy-MM-dd");

    private static final int DATE_POS=0;
    private static final int CLOSE_POS=1;
    private static final int HIGH_POS=2;
    private static final int LOW_POS=3;
    private static final int OPEN_POS=4;
    private static final int VOLUME_POS=5;

    public LocalDate date;
    public double open;
    public double close;
    public double volume;
    public double high;
    public double low;

    public PriceRecord() {
    }

    public PriceRecord(LocalDate date, double close, double high, double low, double open, double volume) {
        this.date = date;
        this.open = open;
        this.close = close;
        this.volume = volume;
        this.high = high;
        this.low = low;
    }

    public static PriceRecord parse(String line) throws Exception {
        String columns [] = line.split("\t");
        PriceRecord priceRecord = new PriceRecord();
        priceRecord.date = formatter.parseLocalDate(columns[DATE_POS]);
        priceRecord.open = new Double(columns[OPEN_POS]);
        priceRecord.close = new Double(columns[CLOSE_POS]);
        priceRecord.volume = new Double(columns[VOLUME_POS]);
        priceRecord.high = new Double(columns[HIGH_POS]);
        priceRecord.low = new Double(columns[LOW_POS]);
        return priceRecord;
    }

    public boolean equals(Object other) {
        if(other == this) {
            return true;
        }

        if(other instanceof PriceRecord) {
            return this.date.equals(((PriceRecord)other).date);
        }
        return false;
    }

    public int hashCode() {
        return date.hashCode();
    }

    public String toString() {
        return String.format("%s\t%f\t%f\t%f\t%f\t%f",
                formatter.print(date),
                close,
                high,
                low,
                open,
                volume);
    }
}