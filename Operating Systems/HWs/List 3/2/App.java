public class App {

    public static void main(String[] args) {
        
        Toalete t = new Toalete();

        Thr1 t1 = new Thr1(t);
        Thr2 t2 = new Thr2(t);

        t1.start();
        t2.start();
        
    }

}