/*
 * EchoClient.java
 */
import java.io.*;
import java.net.*;
import java.util.Scanner;

public class EchoClient {
    
    public static void main(String[] args) {
        
        DatagramSocket datagram;    	// socket de datagrama
        InetAddress address;        	// endereco IP do socket servidor
        DatagramPacket requestPacket;   // pacote sendo enviado
        DatagramPacket responsePacket;  // pacote sendo recebido
        
		Scanner stdIn;       		// leitura de dados
        byte[] buffer;              // buffer de dados
		String message;				// mensagem enviada/recebida
        
		// verifica quantidade de argumentos
        if (args.length != 1){
            System.err.println("Erro: informe o endereço IP do servidor");
            System.exit(0);
        }
            
        try{
			// abre fluxo de entrada de dados via teclado 
			stdIn = new Scanner(System.in);
			
			// abre socket de datagrama
			datagram = new DatagramSocket();

			// le mensagem de requisicao
			System.out.print("Mensagem -> ");
			message = stdIn.nextLine();

			// cria datagrama para armazenar mensagem de requisicao
			buffer = message.getBytes();
			address = InetAddress.getByName(args[0]);
			requestPacket = new DatagramPacket(buffer, buffer.length, address, 1050);

			// envia datagrama contendo mensagem de requisicao
			System.out.println("Envia requisicao...");
			datagram.send(requestPacket);

			// cria datagrama para armazenar mensagem de resposta
			buffer = new byte[256];
			responsePacket = new DatagramPacket(buffer, buffer.length);
			
			// recebe datagrama contendo mensagem de resposta
			System.out.println("Recebe resposta...");
			datagram.receive(responsePacket);

			// imprime mensagem de resposta
			message = new String(responsePacket.getData()).trim();
			System.out.println("Resposta -> " + message);

			// fecha socket de datagrama
			datagram.close();
        } catch(IOException e){
            System.err.println("Erro: " + e);
        }
    }
}
