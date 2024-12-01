/*
 * EchoServer.java
 */
import java.io.*;
import java.net.*;

public class EchoServer {
    
    public static void main(String[] args) {
		// declaracao variaveis
        DatagramSocket datagram;    	// socket de datagrama
        DatagramPacket requestPacket;   // pacote enviado pelo cliente
        DatagramPacket responsePacket;  // pacote contendo a resposta do servidor
		
		InetAddress address;			// endereço IP do socket cliente
		int port;						// numero da porta associada ao socket cliente
        
		byte[] buffer;                 	// buffer de dados
		String message;					// mensagem recebida/enviada
        
        try{
            // abre socket de datagrama
            datagram = new DatagramSocket(1050);
 
			// cria datagrama para armazenar mensagem de requisicao
            buffer = new byte[256];
			requestPacket = new DatagramPacket(buffer, buffer.length);			
			
			// recebe datagrama contendo mensagem de requisicao
            System.out.println("Recebe requisicao...");
            datagram.receive(requestPacket);

			// extrai endereco e nro da porta do socket cliente
			address = requestPacket.getAddress();
			port = requestPacket.getPort();
			
            // extrai mensagem, converte para letras maiuscular e imprime mensagem convertida
            message = new String(requestPacket.getData()).trim().toUpperCase();
            System.out.println(message);

			// cria datagrama contendo mensagem de resposta
            buffer = message.getBytes();
            responsePacket = new DatagramPacket(buffer, buffer.length, address, port);
			
			// envia datagrama contendo mensagem de resposta
			System.out.println("Envia resposta...");
            datagram.send(responsePacket);
 
            // fecha socket de datagrama
            datagram.close();
        } catch(IOException e){
            System.err.println("Erro: " + e);
        }        
    }
}
