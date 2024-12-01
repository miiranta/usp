import java.io.File;

public class PrintSpooler {
    
    String spoolFile;
    File file;
    Sem1 sem = new Sem1(1);

    // construtor classe - recebe como parâmetro o nome do arquivo de spool
    public PrintSpooler(String spoolFile){
        System.out.println("PrintSpooler criado.");

        this.spoolFile = spoolFile;
    }
    
    // método utilizado para abrir o arquivo de spool
    public boolean openPrintSpooler(){

        //Open file
        file = new File(spoolFile);
        
        return true;
    }
    
    // método utilizado para imprimir um job
    public void printJob(String jobName){

        sem.down();

            

        //Print job
        System.out.println("Printing job " + jobName);


    }
    
    // método utilizado para fechar o arquivo de spool
    public void closePrintSpooler(){

        

    }
}