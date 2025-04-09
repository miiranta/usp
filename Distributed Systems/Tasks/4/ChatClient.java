import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class ChatClient {
    
    public static void main(String[] args) {
   	
        ChatInterface obj = null; 
        String username = "Miranda";

        try { 

            Registry registry = LocateRegistry.getRegistry("localhost");

            // Lookup object reference associated to the name "ChatInterface"
            obj = (ChatInterface) registry.lookup("ChatInterface"); 

            // Create a new MessageInterfaceImpl object
            MessageInterfaceImpl messageRef = new MessageInterfaceImpl();

            // Join the chat group with a username and the message reference
            int id = obj.joinGroup(username, messageRef);

            while(true) {
                System.out.println("Enter message to send (or 'exit' to leave): ");
                String msg = System.console().readLine();
                if (msg.equalsIgnoreCase("exit")) {
                    break;
                }
                // Send the message to the chat group
                obj.message(id, msg);
            }

            // Leave the chat group
            obj.leaveGroup(id);
         
        } catch (Exception e) { 
            System.out.println("CalculatorClient exception: " + e.getMessage()); 
            e.printStackTrace(); 
        }        
    }
}
