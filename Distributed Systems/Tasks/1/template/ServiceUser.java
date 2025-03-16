/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
import java.io.*;

public class ServiceUser extends Thread implements ServiceUserInterface{
    // declaring variables
    private ServiceInterface lls;
    private int currentState;

    /**
     * Creates a new instance of ServiceUser
     */
    public ServiceUser(ServiceInterface ref) {
        currentState = 0;
        lls = ref;
    }

    public void result(int valor){
		// write result from calculation

    }

    public void error(){
		// write error message

    }

    public void run(){
		// variable declaration 
		
		// protocol user behaviour

        while (true){
            switch (currentState){
                case 0:
					
					
                    break;
                default:
                    try{
                        sleep(100);
                    } catch(InterruptedException  ie){
                        System.err.println("Erro ao durante a espera pela resposta: "+ie);
                    }
            }
         }
    }
}
