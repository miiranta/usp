/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
import java.io.IOException;
import java.net.*;

public class CalculatorProtocolClient extends Thread implements ServiceInterface{
    
    // declaring constant
    private final short TIMEOUT = 100;

    // declaring variables
    private ServiceUserInterface serviceUser;	// Service user reference
    private int currentState;                 	// Protocol entity current state

    private DatagramSocket datagram;            // Socket datagram
    private InetAddress remoteAddress;          // Remote address
    private int remotePortNumber;               // remote port number

    private RequestPDU pdu = null;  			// Request PDU

    /** Creates a new instance of UnicastProtocol */
    public CalculatorProtocolClient( int portNumber, String serverAddr, int serverPortNumber) {

        currentState = 0;
        remotePortNumber = serverPortNumber;

        try{
            remoteAddress = InetAddress.getByName(serverAddr);
            datagram = new DatagramSocket(portNumber);
        }catch(IOException se){
            System.err.println("Nao foi possivel inicializar protocolo: "+se);
            System.exit(0);
        }
    }

    public void setRef(ServiceUserInterface user){
        serviceUser = user;
    }

    public void add(int op1, int op2){
        DatagramPacket requestPacket;

        // <RSPPDU><espaço><respcode><espaço><resultado>
        // espcode é um número inteiro que representa sucesso (ou não) da realização da operação (0 – falha e 1 – sucesso);

        // create pdu
        try {
            pdu = new RequestPDU(0, op1, op2);
        } catch (Exception e){
            System.err.println(e);
            System.exit(0);
        }

        // create packet
        requestPacket = new DatagramPacket(pdu.getPDUData(), pdu.getPDUData().length, remoteAddress, remotePortNumber);

        // send packet
        try{
            datagram.send(requestPacket);
        } catch (IOException ioe){
            System.err.println("Could not send data: "+ioe);
            System.exit(0);
        }
        
        // update current state
        currentState = 1;
    }

    public void sub(int op1, int op2){
        DatagramPacket requestPacket;

        // create pdu
        try {
            pdu = new RequestPDU(1, op1, op2);
        } catch (IllegalFormatException e){
            System.err.println(e);
            System.exit(0);
        }

        // create packet
        requestPacket = new DatagramPacket(pdu.getPDUData(), pdu.getPDUData().length, remoteAddress, remotePortNumber);

        // send packet
        try{
            datagram.send(requestPacket);
        } catch (IOException ioe){
            System.err.println("Could not send data: "+ioe);
            System.exit(0);
        }

        // update current state
        currentState = 1;       
    }
    
    @SuppressWarnings("override")
    public void times(int op1, int op2){
        DatagramPacket requestPacket;

        // create pdu
        try {
            pdu = new RequestPDU(2, op1, op2);
        } catch (IllegalFormatException e){
            System.err.println(e);
            System.exit(0);
        }

        // create packet
        requestPacket = new DatagramPacket(pdu.getPDUData(), pdu.getPDUData().length, remoteAddress, remotePortNumber);

        // send packet
        try{
            datagram.send(requestPacket);
        } catch (IOException ioe){
            System.err.println("Could not send data: "+ioe);
            System.exit(0);
        }

        // update current state
        currentState = 1;   
    }

    @SuppressWarnings("override")
    public void div(int op1, int op2){
        DatagramPacket requestPacket;

        // create pdu
        try {
            pdu = new RequestPDU(3, op1, op2);
        } catch (IllegalFormatException e){
            System.err.println(e);
            System.exit(0);
        }

        // create packet
        requestPacket = new DatagramPacket(pdu.getPDUData(), pdu.getPDUData().length, remoteAddress, remotePortNumber);

        // send packet
        try{
            datagram.send(requestPacket);
        } catch (IOException ioe){
            System.err.println("Could not send data: "+ioe);
            System.exit(0);
        }

        // update current state
        currentState = 1;   
    }

    public void run(){
        DatagramPacket requestPacket = null;    // Datagram for receiving a calculation request
        DatagramPacket responsePacket = null;   // Datagram for sending a calculation response

        byte[] buf;                             // Buffer used to store data
        ResponsePDU respPdu = null;             // ResponsePDU

        try {
            datagram.setSoTimeout(TIMEOUT);
        } catch (SocketException se){
            System.err.println("Could not set timeout");
            System.exit(0);
        }

        while (true){ // check for incoming packets from network
             // Ckeck protocol current state
            switch (currentState){
                case 0:
					// sleep
                    try{
                        Thread.sleep(1000);
                    } catch (InterruptedException ie){
                        System.err.println("Thread interrupted: "+ie);
                    }
                    break;
                case 1:
                    buf = new byte[128];

                    // try receive Response PDU
                    try{
                        // set timer
                        responsePacket = new DatagramPacket(buf, buf.length);
                        datagram.receive(responsePacket);

                        // extracts pdu
                        try {
                            respPdu = new ResponsePDU(responsePacket.getData());
                        } catch (Exception e){
                            System.err.println(e);
                            continue;
                        }
                        
                        // check response
                        if(respPdu.getRespcode() == 1){
                            // success
                            serviceUser.result(respPdu.getResult());
                        } else {
                            // failure
                            serviceUser.error();
                        }
                        
                        // update state
                        currentState = 0;
                    } catch (SocketException se){
                            System.err.println("Could not set timeout");
                    } catch(SocketTimeoutException ste){
                        // timeout - retransmit
                        // create packet
                        requestPacket = new DatagramPacket(pdu.getPDUData(), pdu.getPDUData().length, remoteAddress, remotePortNumber);
                        
                        // send packet
                        try{
                            datagram.send(requestPacket);
                        } catch (IOException ioe){
                            System.err.println("Could not send data: "+ioe);
                        }

                    } catch (IOException ioe){
                        System.err.println("Could not receive data: "+ioe);
                    }
                    break;
            } // switch
        } //while
    }
}