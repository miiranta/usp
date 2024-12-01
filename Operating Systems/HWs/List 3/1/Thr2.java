public class Thr2 extends Thread {

    private Sem1 sem;

    public Thr2(Sem1 sem) {
        this.sem = sem;
    }

    public void run() {
        sem.down();
        System.out.println("Thr2");
        sem.up();
    }

}