package apps.stockprediction;

import org.joda.time.LocalDate;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PriceRecordDB {

    private final File db;
    private final List<PriceRecord> priceRecords = new ArrayList<>();
    private static final String HEADER = "DATE\tCLOSE\tHIGH\tLOW\tOPEN\tVOLUME";

    public PriceRecordDB(String dir, String name) throws Exception {
        db = new File(dir, name);
        read();
    }

    public List<PriceRecord> get() {
        return get(null);
    }

    public List<PriceRecord> get(LocalDate fromDate) {
        List<PriceRecord> result = new ArrayList<>();

        for(PriceRecord priceRecord : this.priceRecords) {
            if(fromDate != null && priceRecord.date.isBefore(fromDate)) {
                continue;
            }
            result.add(priceRecord);
        }
        return result;
    }

    private void read() throws Exception {
        if(!db.exists()) {
            return;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(db)));

        String line;
        boolean header = true;
        while((line = br.readLine()) != null) {
            if(header) {
                header = false;
                continue;
            }

            priceRecords.add(PriceRecord.parse(line));
        }
    }

    public void add(List<PriceRecord> priceRecords) {
        for(PriceRecord priceRecord : priceRecords) {
            if(!this.priceRecords.contains(priceRecord)) {
                this.priceRecords.add(priceRecord);
            }
        }

        Collections.sort(this.priceRecords, (d1, d2) -> d1.date.compareTo(d2.date));
    }

    public List<PriceRecord> getPriceRecords() {
        return priceRecords;
    }

    public void write() throws Exception {
        FileOutputStream out = new FileOutputStream(db);
        out.write((HEADER + "\n").getBytes());
        for(PriceRecord priceRecord : priceRecords) {
            out.write((priceRecord.toString() + "\n").getBytes());
        }
    }
}
