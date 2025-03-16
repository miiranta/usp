/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

public class ResponsePDU {
    // Data declaration
    private String pduData;
    private int respcode;
    private int value;

    /** Creates a new instance of ResponsePDU */
    public ResponsePDU(int respcode, int valor) throws IllegalFormatException{

        // check opcode
        if (respcode < 0 || respcode > 1){ // error
            throw new IllegalFormatException();
        } else{
            // create pdu
			
        }
    }

    /** Creates a new instance of ResponsePDU from an array of bytes */
    public ResponsePDU(byte[] data) throws IllegalFormatException{
		String[] elements;
		
        pduData = new String(data);

        // parse PDU code
        elements = pduData.split(" ");
        
		// check pdu format
		
    }

    public int getRespcode(){
        return respcode;
    }

    public int getResult(){
        return value;
    }

    public byte[] getPDUData(){
        return pduData.getBytes();
    }
}
