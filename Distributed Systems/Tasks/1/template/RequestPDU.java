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

    /*
    <RQTPDU><espaço><opcode><espaço><operando1><espaço><operando2> 
    Onde:  
    opcode é um número inteiro que representa o código da operação (0 – soma; 1 – subtração; 2 – 
    multiplicação; e  3 – divisão); 
    operando1 e operando2 são números inteiros;
    */

    /** Creates a new instance of RequestPDU */
    public RequestPDU(int opcode, int op1, int op2) throws IllegalFormatException{

        // check opcode
        if (opcode < 0 || opcode > 3){ // error
            throw new IllegalFormatException();
        } else{
            // create new pdu
            pduData = new String("RQTPDU " + opcode + " " + op1 + " " + op2);
            this.opcode = opcode;
            this.op1 = op1;
            this.op2 = op2;
        }
    }

    /** Creates a new instance of RequestPDU from an array of bytes */
    public RequestPDU(byte[] data) throws IllegalFormatException{
		
        String[] elements;

        pduData = new String(data);

        // parse PDU 
        elements = pduData.split(" ");

		// check pdu format
        string RQTPDUstr = elements[0];
        if (!RQTPDUstr.equals("RQTPDU")){
            throw new IllegalFormatException();
        } 

        opcode = Integer.parseInt(elements[1]);
        if (opcode < 0 || opcode > 3){
            throw new IllegalFormatException();
        } 

        op1 = Integer.parseInt(elements[2]);
        op2 = Integer.parseInt(elements[3]);
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
