package apps.stockprediction;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class TestPredictor {
    @Test
    public void analysePhia() throws Exception {
        String exchange = "AMS";
        String stock = "PHIA";

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(String.format("%s-%s-avg", exchange, stock))));

        String line;
        boolean header = true;
        double previousClose=0;

        while((line = br.readLine()) != null) {
            if(header) {
                header = false;
                continue;
            }

            String [] columns = line.split("\t");

            double close = new Double(columns[1]);

            if(previousClose !=0) {
                System.out.println("Delta: " + (((close/previousClose) - 1)) * 100);
            }

            previousClose = close;
        }
    }
}
