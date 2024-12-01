/*
 * EchoServer.java
 */
import java.net.*;
import java.io.*;

public class EchoServer {
    
    public static void main(String[] args) {
        
		// declaracao das variaveis
        ServerSocket serverSocket;      // Socket TCP servidor (conexao)
        Socket clientSocket;            // Socket TCP de comunicacao com o cliente
        String message;               	// Mensagem enviada pelo cliente
		BufferedReader in;				// Entrada(recepcao) formatada de dados
		PrintWriter out;				// Saida (envio) formatado de dados

        try {
            // abre socket TCP servidor
            serverSocket = new ServerSocket(1050);

            // espera por requisicao de conexao enviada pelo socket cliente
            clientSocket = serverSocket.accept();	// cria socket TCP de comunicacao com o cliente

            // abre fluxos de entrada e saida de dados associados ao socket TCP de comunicacao com o cliente
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            out = new PrintWriter(clientSocket.getOutputStream(), true);
            
			// recebe mensagem de requisicao, escreve mensagem de requisicao e envia mensagem de resposta		
            System.out.println("Servidor pronto...");
            message = in.readLine();
            if (message != null) {
                System.out.println("Do cliente ->" + message);
                out.println(message.toUpperCase());		// envia mensagem de volta para o cliente
            }
                
            // fecha fluxos de entrada e saida de dados
            out.close();
            in.close();

            // fecha sockets TCP de comunicacao com o cliente e TCP servidor
            clientSocket.close();
            serverSocket.close();      
        } catch (IOException e) {
            System.err.println("Erro: " + e);
        }             
    }
}
