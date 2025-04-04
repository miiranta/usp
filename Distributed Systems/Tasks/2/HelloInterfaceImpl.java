import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class HelloInterfaceImpl extends UnicastRemoteObject implements HelloInterface {

	public HelloInterfaceImpl() throws RemoteException {
		super();
	}
	
	public String sayHello() throws RemoteException{
		System.out.println("Call!!");
		return "Hello World!";	
	}
}