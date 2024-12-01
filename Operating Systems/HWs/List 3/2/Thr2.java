public class Thr2 extends Thread {

    private Toalete t;

    public Thr2(Toalete t) {
        this.t = t;
    }

    public void run() {
       
        t.homemQuerEntrar("Homem 1");
        t.homemQuerEntrar("Homem 2");
        t.homemQuerEntrar("Homem 3");
        t.homemQuerEntrar("Homem 4");
        t.homemQuerEntrar("Homem 5");
        t.homemQuerEntrar("Homem 6");

        t.homemSaiToalete("Homem 1");
        t.homemSaiToalete("Homem 2");
        t.homemSaiToalete("Homem 3");
        t.homemSaiToalete("Homem 4");
        t.homemSaiToalete("Homem 5");
        t.homemSaiToalete("Homem 6");

    }

}