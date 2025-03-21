/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
import java.util.NoSuchElementException;
import java.util.Scanner;

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
        System.out.println("Resultado: "+valor);
        currentState = 0;
    }

    public void error(){
		// write error message
        System.out.println("Erro na operacao");
        currentState = 0;
    }

    public void run(){

        // ADD 10 5
        // SUB 10 5
        // MUL 10 5
        // DIV 10 5

		// variable declaration 
        Scanner sc;
        String operation;
        String[] servicePrimitive;
        int op1, op2;
		
		// protocol user behaviour
        sc = new Scanner(System.in);
        System.out.println("Digite a operacao (ADD, SUB, MUL, DIV) e os operandos (ex: ADD 10 5): (or EXIT to finish)");
                        
        while (true){
            switch (currentState){
                case 0:
                    try {
                        System.out.print("--> ");
                        operation = sc.nextLine();

                        servicePrimitive = operation.split(" ");

                        if (servicePrimitive.length == 1 && servicePrimitive[0].equals("EXIT")){
                            System.exit(0);
                        }

                        if (servicePrimitive.length != 3){
                            System.out.println("Erro: formato invalido");
                            continue;
                        }

                        try{
                            op1 = Integer.parseInt(servicePrimitive[1]);
                            op2 = Integer.parseInt(servicePrimitive[2]);
                        } catch (NumberFormatException nfe){
                            System.out.println("Erro: formato invalido");
                            continue;
                        }

                        switch(servicePrimitive[0]){
                            case "ADD":
                                lls.add(op1, op2);
                                currentState = 1;
                                System.out.println("Aguardando resposta...");
                                break;
                            case "SUB":
                                lls.sub(op1, op2);
                                currentState = 1;
                                System.out.println("Aguardando resposta...");
                                break;
                            case "MUL":
                                lls.times(op1, op2);
                                currentState = 1;
                                System.out.println("Aguardando resposta...");
                                break;
                            case "DIV":
                                lls.div(op1, op2);
                                currentState = 1;
                                System.out.println("Aguardando resposta...");
                                break;
                            default:
                                System.out.println("Erro: operacao invalida");
                                continue;
                        }

                    } catch (NoSuchElementException nsee){
                        System.err.println("Erro durante a leitura da operacao: "+nsee);
                    }

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
