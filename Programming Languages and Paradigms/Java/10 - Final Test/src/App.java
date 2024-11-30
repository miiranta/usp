public class App {
    public static void main(String[] args) throws Exception {
        
        //Impossible to change value, its like a constant
        //Final.b = "lol";
        System.out.println(Final.b);


        //Finals can be declared in the constructor
        Const c = new Const(10);
        System.out.println(c.FINAL_IN_CONSTRUCTOR);
    }
}
