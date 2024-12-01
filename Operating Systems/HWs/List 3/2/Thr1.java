public class Thr1 extends Thread {

    private Toalete t;

    public Thr1(Toalete t) {
        this.t = t;
    }

    public void run() {
       
        t.mulherQuerEntrar("Mulher 1");
        t.mulherQuerEntrar("Mulher 2");
        t.mulherQuerEntrar("Mulher 3");
        t.mulherQuerEntrar("Mulher 4");
        t.mulherQuerEntrar("Mulher 5");
        t.mulherQuerEntrar("Mulher 6");

        t.mulherSaiToalete("Mulher 1");
        t.mulherSaiToalete("Mulher 2");
        t.mulherSaiToalete("Mulher 3");
        t.mulherSaiToalete("Mulher 4");
        t.mulherSaiToalete("Mulher 5");
        t.mulherSaiToalete("Mulher 6");

    }

}