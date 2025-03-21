/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

public class CalculatorServer {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        int port;
        CalculatorProtocolServer server;

        try{
			// extracts port number
            port = Integer.parseInt(args[0]);

			// create server entity
            server = new CalculatorProtocolServer(port);
            server.start();

        } catch (NumberFormatException nfe){
            System.err.println("Erro formato nro porta");
        }
    }

}
