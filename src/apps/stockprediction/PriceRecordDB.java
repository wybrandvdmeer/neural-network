package apps.stockprediction;

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
            priceRecords.add(PriceRecord.parse(line));
        }

        return priceRecords;
    }

    public void add(List<PriceRecord> priceRecords) {
        for(PriceRecord priceRecord : priceRecords) {
            if(!this.priceRecords.contains(priceRecord)) {
                this.priceRecords.add(priceRecord);
            }
        }

        Collections.sort(this.priceRecords, (d1, d2) -> (int)(d1.date.getTime() - d2.date.getTime()));
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
