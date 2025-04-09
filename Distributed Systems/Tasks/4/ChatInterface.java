import java.rmi.Remote;
import java.rmi.RemoteException;

public interface ChatInterface extends Remote{
    public int joinGroup(String name, MessageInterface ref) throws RemoteException;
    public void leaveGroup(int id) throws RemoteException;
    public void message(int id, String msg) throws RemoteException;
}
