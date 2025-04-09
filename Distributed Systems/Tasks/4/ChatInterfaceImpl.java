import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashMap;

public class ChatInterfaceImpl extends UnicastRemoteObject implements ChatInterface {

    // Array of MessageInterface references
    private HashMap<Integer, MessageInterface> messageRefs = new HashMap<Integer, MessageInterface>();
    private HashMap<Integer, String> userNames = new HashMap<Integer, String>();

    ChatInterfaceImpl() throws RemoteException {
        super();
    }

    public int joinGroup(String name, MessageInterface ref) throws RemoteException {
        
        // Create random user ID
        int id = (int) (Math.random() * 1000);

        // Add the reference to the list
        while(messageRefs.containsKey(id)) {
            // If it does, generate a new ID
            id = (int) (Math.random() * 1000);
        }

        // Check if username is already taken
        for (String existingName : userNames.values()) {
            if (existingName.equals(name)) {
                throw new RemoteException("Username already taken");
            }
        }

        messageRefs.put(id, ref);
        userNames.put(id, name);
        
        return id;
    }

    public void leaveGroup(int id) throws RemoteException {
        messageRefs.remove(id);
        userNames.remove(id);
    }

    public void message(int id, String msg) throws RemoteException {
        // Send the message to all users
        for (MessageInterface ref : messageRefs.values()) {
            try {
                ref.messageNotification(userNames.get(id), msg);
            } catch (RemoteException e) {
                // Handle the exception
                System.out.println("Error sending message to user: " + e.getMessage());
            }
        }
    }

}