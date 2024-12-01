public class Sem1 {

    private int resorce;

    public Sem1(int a) {

        if(a <= 0) {
            throw new IllegalArgumentException("Argument must be an int > 0");
        }

        this.resorce = a;
    }

    public synchronized void down() {

        while(resorce == 0) {
            try{
                System.out.println("Waiting");

                wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        resorce--;
    }

    public synchronized void up() {
        resorce++;
        notify();
    }

}