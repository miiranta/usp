import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class CalculatorClient {
    
    public static void main(String[] args) {
   	
        CalculatorInterface obj = null; 

        try { 

	    Registry registry = LocateRegistry.getRegistry("192.168.1.89");

	    // Lookup object reference associated to the name "CalculatorServer"
            obj = (CalculatorInterface)registry.lookup("CalculatorServer"); 

            String message = Integer.toString(obj.add(10, 20));
            System.out.println("CalculatorClient 10 + 20: " + message);

            message = Integer.toString(obj.sub(10, 20));
            System.out.println("CalculatorClient 10 - 20: " + message);

            message = Integer.toString(obj.times(10, 20));
            System.out.println("CalculatorClient 10 * 20: " + message);

            message = Integer.toString(obj.div(10, 20));
            System.out.println("CalculatorClient 10 / 20: " + message);

        } catch (Exception e) { 
            System.out.println("CalculatorClient exception: " + e.getMessage()); 
            e.printStackTrace(); 
        }        
    }
}
