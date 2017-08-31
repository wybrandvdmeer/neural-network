package apps.sourcedetector;

import java.io.File;
import java.util.Arrays;

public class SourceClassifier {
    public static void main(String [] args) throws Exception {

        Arrays.stream(args).forEach( arg -> {
            File sourceDir = new File(args[0]);

            if(!sourceDir.isDirectory()) {
                throw new RuntimeException(String.format("Argument %s is not a directory.", args[0]));
            }

            try {
                processFiles(sourceDir);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    private static void processFiles(File root) throws Exception {
        for(File file : root.listFiles()) {
            if(file.isDirectory()) {
                processFiles(file);
            } else {
                String extenstion = getExtension(file.getName());

                boolean isJava = "java".equals(extenstion);
                boolean isPython = "py".equals(extenstion);

                System.out.println(String.format("File: %s", file.getName()));

                KeywordCounter keywordCounter = new KeywordCounter(file);
            }
        }
    }

    private static String getExtension(String file) {
        int index;
        if((index = file.lastIndexOf(".")) < 0) {
            return null;
        }
        return file.substring(index + 1, file.length());
    }
}

