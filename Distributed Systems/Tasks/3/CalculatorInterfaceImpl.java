import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class CalculatorInterfaceImpl extends UnicastRemoteObject implements CalculatorInterface {

	public CalculatorInterfaceImpl() throws RemoteException {
		super();
	}
	
	public int add(int a, int b) throws RemoteException {
        return a + b;
    }

    public int sub(int a, int b) throws RemoteException {
        return a - b;
    }

    public int times(int a, int b) throws RemoteException {
        return a * b;
    }

    public int div(int a, int b) throws RemoteException {
        if (b == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return a / b;
    }
}