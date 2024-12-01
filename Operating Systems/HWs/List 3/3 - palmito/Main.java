
import java.io.File;

public class Main {
    public static void main(String[] args) {
        PrintSpooler printSpooler = new PrintSpooler("C:\\Users\\10716845\\Documents\\PrintSpooler\\src\\spool.txt");

        // Abre o arquivo de spool
        printSpooler.openPrintSpooler();

        // Caminho para a pasta que contém os arquivos de job
        String jobFolderPath = "C:\\Users\\10716845\\Documents\\PrintSpooler\\src\\files";

        // Localiza todos os arquivos de job na pasta
        File jobFolder = new File(jobFolderPath);
        File[] jobFiles = jobFolder.listFiles();

        // Verifica se há arquivos de job encontrados
        if (jobFiles != null && jobFiles.length > 0) {
            // Cria e inicia as threads responsáveis pela submissão das requisições de
            // impressão
            int numThreads = jobFiles.length; // Número de threads igual ao número de arquivos de job
            Thread[] threads = new Thread[numThreads];

            for (int i = 0; i < numThreads; i++) {
                final String jobPath = jobFiles[i].getAbsolutePath();
                threads[i] = new Thread(() -> {
                    try {
                        Thread.sleep((long) (Math.random() * 5000)); // Espera um período de tempo aleatório

                        printSpooler.printJob(jobPath);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                });
                threads[i].start();
            }

            // Aguarda todas as threads concluírem
            for (int i = 0; i < numThreads; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        } else {
            System.out.println("Nenhum arquivo de job encontrado na pasta: " + jobFolderPath);
        }

        // Fecha o arquivo de spool
        printSpooler.closePrintSpooler();
    }
}