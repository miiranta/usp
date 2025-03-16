/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

public class RequestPDU {
    // Data declaration
    private String pduData;
    private int opcode;
    private int op1;
    private int op2;

    /** Creates a new instance of RequestPDU */
    public RequestPDU(int opcode, int op1, int op2) throws IllegalFormatException{

        // check opcode
        if (opcode < 0 || opcode > 3){ // error
            throw new IllegalFormatException();
        } else{
            // create new pdu
        }
    }

    /** Creates a new instance of RequestPDU from an array of bytes */
    public RequestPDU(byte[] data) throws IllegalFormatException{
		String[] elements;
		
        pduData = new String(data);

        // parse PDU 
        elements = pduData.split(" ");
        
		// check pdu format
    }

    public int getOpcode(){
        return opcode;
    }

    public int getOp1(){
        return op1;
    }

    public int getOp2(){
        return op2;
    }

    public byte[] getPDUData(){
        return pduData.getBytes();
    }
}
