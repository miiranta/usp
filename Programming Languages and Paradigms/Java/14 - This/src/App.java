public class App {
    public static void main(String[] args) throws Exception {

    //Scope Test
        ThisTest t = new ThisTest();
        t.shadow();

        System.out.println("");

    //this() and constructor overloading test
        new Constructor();

    }
}
