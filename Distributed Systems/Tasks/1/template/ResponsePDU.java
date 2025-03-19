/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

public class ResponsePDU {
    // Data declaration
    private String pduData;
    private int respcode;
    private int value;

    /*
    Codificação concreta: 
    <RSPPDU><espaço><respcode><espaço><resultado> 
    Onde:  
    respcode é um número inteiro que representa sucesso (ou não) da realização da operação (0 – 
    falha e 1 – sucesso); 
    resultado é um número inteiro; 
    */

    /** Creates a new instance of ResponsePDU */
    public ResponsePDU(int respcode, int valor) throws IllegalFormatException{

        // check opcode
        if (respcode < 0 || respcode > 1){ // error
            throw new IllegalFormatException();
        } else{
            // create pdu
            pduData = new String("RSPPDU " + respcode + " " + valor);
            this.respcode = respcode;
            this.value = valor;
        }
    }

    /** Creates a new instance of ResponsePDU from an array of bytes */
    public ResponsePDU(byte[] data) throws IllegalFormatException{
		String[] elements;
		
        pduData = new String(data);

        // parse PDU code
        elements = pduData.split(" ");
        
		// check pdu format
        String RSPPDUstr = elements[0];
        if(!RSPPDUstr.equals("RSPPDU")){
            throw new IllegalFormatException();
        }

        respcode = Integer.parseInt(elements[1]);
        if (respcode < 0 || respcode > 1){
            throw new IllegalFormatException();
        }

        value = Integer.parseInt(elements[2]);
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
