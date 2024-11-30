public class Main {
    public static void main(String[] args) throws Exception {
        
        //Cannot call A directly
        //new A();

        //Calls using subclass
        A b = new B();
        b.b();
        
    }
}
