/*
 * EchoClient.java
 */
import java.util.Scanner;
import java.io.*;
import java.net.*;

public class EchoClient {
   
    public static void main(String[] args) {

		// declaracao das variaveis
        Socket echoSocket;          // Socket TCP cliente
		BufferedReader in;			// Entrada(recepcao) formatada de dados
		PrintWriter out;			// Saida (envio) formatado de dados
        Scanner stdIn;       		// Fluxo de entrada de dados via teclado
        String message;           	// Mensagem do usuario

		// verifica quantidade de argumentos
        if (args.length != 1){
            System.err.println("Erro: informe o endereço IP do servidor");
            System.exit(0);
        }
        
        try {
			// abre fluxo de entrada de dados via teclado 
			stdIn = new Scanner(System.in);
			
            // abre socket TCP cliente na porta 1050
            echoSocket = new Socket(args[0], 1050);

            // abre fluxos de entrada e saida de dados para o socket
            out = new PrintWriter(echoSocket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(echoSocket.getInputStream()));
	
            // le mensagem de requisicao, envia mensagem de requisicao e recebe mensagem de resposta
            System.out.print("Para servidor -> ");
            message = stdIn.nextLine();
            if (message != null){
                out.println(message);
                System.out.println("Mensagem de echo: " + in.readLine());
            }

            // fecha fluxos de entrada e saida de dados
            out.close();
            in.close();
            stdIn.close();

            // fecha socket TCP cliente
            echoSocket.close();      
        } catch (IOException e) {
            System.err.println("Erro: " + e);
        }         
    }    
}
