package apps.sourcedetector;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;

public class KeywordCounter {

    public static String [] keywords = { "public", "protected", "private", "package", "import", "class", "java", "false", "False", "true", "True", "def ", "{", "(", ":" };
    public int [] keywordOccurrences = new int[keywords.length];

    public static int getNoOfKeywords() {
        return keywords.length;
    }

    public KeywordCounter(File input) throws Exception {
        Arrays.sort(keywords, ((o1, o2) -> o2.length() - o1.length()));
        countKeywordOccurrences(new FileInputStream(input));
    }

    private void countKeywordOccurrences(InputStream inputStream) throws Exception {

        int maxLength = keywords[0].length();

        Fifo fifo = new Fifo(maxLength);

        int token;

        while((token = inputStream.read()) != -1) {

            fifo.put(token);

            for (int idx = 0; idx < keywords.length; idx++) {
                int length = keywords[idx].length() < fifo.getLength() ? keywords[idx].length() : fifo.getLength();

                if (keywords[idx].equals(fifo.toString().substring(0, length))) {
                    keywordOccurrences[idx]++;
                }
            }
        }

        for(int idx=1; idx < maxLength && idx < fifo.toString().length(); idx++) {
            for (int idx2 = 0; idx2 < keywords.length; idx2++) {
                if (keywords[idx2].equals(fifo.toString().substring(idx))) {
                    keywordOccurrences[idx]++;
                }
            }
        }
    }

    public int getKeywordOccurrence(String keyword) throws Exception {
        for(int idx=0; idx < keywords.length; idx++) {
            if(keyword.equals(keywords[idx])) {
                return keywordOccurrences[idx];
            }
        }

        throw new Exception(String.format("Keyword %s is not counted.", keyword));
    }
}
