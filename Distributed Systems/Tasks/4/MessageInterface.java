import java.rmi.Remote;
import java.rmi.RemoteException;

public interface MessageInterface extends Remote{
    void messageNotification(String sender, String message) throws RemoteException;
}
