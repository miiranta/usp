/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
import java.io.IOException;
import java.net.*;

public class CalculatorProtocolServer extends Thread{

    // declaring variables
    private DatagramSocket datagram;            // Socket datagram

    /** Creates a new instance of CalculatorProtocolServer */
    public CalculatorProtocolServer( int portNumber) {
        try{
            datagram = new DatagramSocket(portNumber);
        }catch(IOException se){
            System.err.println("Nao foi possivel inicializar protocolo: "+se);
            System.exit(0);
        }
    }

    public void run(){
        DatagramPacket requestPacket = null;    // Datagram for receiving a calculation request
        DatagramPacket responsePacket = null;   // Datagram for sending a calculation response

        byte[] buf;                             // Buffer used to store data
        ResponsePDU respPdu = null;             // ResponsePDU
        RequestPDU reqPdu = null;               // RequestPDU
        int op1, op2, result = 0, respCode = 0;

        while (true){ // check for incoming packets from network

            buf = new byte[128];

			// try receive request PDU
            try{ 
                requestPacket = new DatagramPacket(buf, buf.length);
                datagram.receive(requestPacket);

                // extracts pdu
                

                // extracts info
                
                
                // check request
                switch(reqPdu.getOpcode()){
                    case 0: // calculate add
                        
                        break;
                    case 1: // calculate sub
                        
                        break;
                    case 2: // calculate times
                        
                        break;
                    case 3: // calculate div
                        
                        break;
                }

                // create response PDU
               

                // create response packet
               

                // send response packet
                
            } catch (IOException ioe){
                System.err.println("Could not receive data: "+ioe);
            }
        } //while
    }
}