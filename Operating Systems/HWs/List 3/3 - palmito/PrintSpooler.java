import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class PrintSpooler {

    private String spoolFile;
    private Semaphore semaphore;
    private FileWriter fileWriter;

    // construtor classe - recebe como parâmetro o nome do arquivo de spool
    public PrintSpooler(String spoolFile) {
        this.spoolFile = spoolFile;
        this.semaphore = new Semaphore(1); // Inicializa o semáforo com 1 recurso (impressora disponível)

    }

    // método utilizado para abrir o arquivo de spool
    public boolean openPrintSpooler() {
        try {
            fileWriter = new FileWriter(spoolFile);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    // Método utilizado para imprimir um job
    public void printJob(String jobName) {
        semaphore.down(); // Aguarda uma impressora disponível
        try {
            // Simula a impressão, copiando o conteúdo do arquivo de job para o arquivo de
            // spool
            BufferedReader jobReader = new BufferedReader(new FileReader(jobName));

            String line;
            fileWriter.write("Novo job: "+jobName + "\n"); // Adiciona o nome do job ao arquivo de spool
            while ((line = jobReader.readLine()) != null) {
                // Copia a linha para o arquivo de spool
                fileWriter.write(line + "\n");
            }

            jobReader.close();
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            semaphore.up(); // Libera a impressora para que outro job possa ser impresso
        }
    }

    // Método utilizado para fechar o arquivo de spool
    public void closePrintSpooler() {
        try {
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
