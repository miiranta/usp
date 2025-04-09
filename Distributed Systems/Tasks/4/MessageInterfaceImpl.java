import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class MessageInterfaceImpl extends UnicastRemoteObject implements MessageInterface {

    MessageInterfaceImpl() throws RemoteException {
        super();
    }

    public void messageNotification(String sender, String message) throws RemoteException {
        System.out.println("Message from " + sender + ": " + message);
    }

}