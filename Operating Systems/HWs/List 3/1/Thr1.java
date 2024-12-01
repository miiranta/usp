public class Thr1 extends Thread {

    private Sem1 sem;

    public Thr1(Sem1 sem) {
        this.sem = sem;
    }

    public void run() {
        sem.down();
        System.out.println("Thr1");
        sem.up();
    }

}