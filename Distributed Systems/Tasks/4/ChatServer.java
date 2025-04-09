import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class ChatServer {
    
    public static void main(String[] args) {
        
        try {
            MessageInterfaceImpl obj = new MessageInterfaceImpl();

            // Bind this object instance to the name "ChatServer"
            Registry registry = LocateRegistry.getRegistry("localhost");
            registry.rebind("ChatInterface", obj);

            System.out.println("ChatServer bound in registry");
        } catch (RemoteException e) {
            System.out.println("ChatServer err: " + e.getMessage());
        }
    }
    
}
