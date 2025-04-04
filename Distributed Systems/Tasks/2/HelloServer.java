import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class HelloServer {
    
    public static void main(String[] args) {
        
        try {
            HelloInterfaceImpl obj = new HelloInterfaceImpl();

            // Bind this object instance to the name "HelloServer"
            Registry registry = LocateRegistry.getRegistry("localhost");
            registry.rebind("HelloServer", obj);

            System.out.println("HelloServer bound in registry");
        } catch (RemoteException e) {
            System.out.println("HelloServer err: " + e.getMessage());
        }
    }
    
}


            