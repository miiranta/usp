package streams;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class MachineDataConverterOutputStream {
    private File stream;
    FileWriter streamWriter;
    private final String pathPrefix = "../outputFiles/";

    public MachineDataConverterOutputStream(String fileName) {
        stream = new File(pathPrefix + fileName);
    }

    public void contentToFile(String content) {
        try {
            streamWriter = new FileWriter(stream.getPath());
            streamWriter.write(content);
            streamWriter.close();

            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("It wasn't possible to write to file...");
            e.printStackTrace();
        }
    }
}
