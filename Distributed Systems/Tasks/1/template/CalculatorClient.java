/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

public class CalculatorClient{

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        int localPort, remotePort;
        String remoteAddress;
        CalculatorProtocolClient client;
        ServiceUser user;

        // check number of arguments
        if (args.length == 3){
            try{
			    // parse arguments
                localPort = Integer.parseInt(args[0]);
                remoteAddress = args[1];
                remotePort = Integer.parseInt(args[2]);

                // create protocol entity
                client = new CalculatorProtocolClient(localPort, remoteAddress, remotePort);

				// create service user 
                user = new ServiceUser(client);

                // start entities
                client.start();
                user.start();

            } catch (NumberFormatException nfe){
                System.err.println("Erro formato nro porta");
            }
        } else{
            System.err.println("Erro nro argumentos: java CalculatorClient <localPort> <remoteAddress> <remotePort>");
        }

    }

}
