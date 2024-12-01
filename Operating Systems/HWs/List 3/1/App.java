public class App {

    public static void main(String[] args) {
        
        Sem1 sem = new Sem1(1);

        Thr1 thr1 = new Thr1(sem);
        Thr2 thr2 = new Thr2(sem);
        thr1.start();
        thr2.start();
        try {
            thr1.join();
            thr2.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


    }

}