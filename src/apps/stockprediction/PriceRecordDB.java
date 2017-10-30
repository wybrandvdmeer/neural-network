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

    public PriceRecordDB(String dir, String name) {
        db = new File(dir, name);
    }

    public List<PriceRecord> read() throws Exception {
        return read(null);
    }

    public List<PriceRecord> read(LocalDate fromDate) throws Exception {
        if(!db.exists()) {
            return null;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(db)));

        String line;
        boolean header = true;
        while((line = br.readLine()) != null) {
            if(header) {
                header = false;
                continue;
            }
            PriceRecord priceRecord = PriceRecord.parse(line);

            if(fromDate != null && fromDate.isBefore(priceRecord.date)) {
                continue;
            }

            priceRecords.add(priceRecord);
        }

        return priceRecords;
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
