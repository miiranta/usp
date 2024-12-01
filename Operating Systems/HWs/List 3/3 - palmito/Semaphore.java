class Semaphore {

    private int resources;

    public Semaphore(int resources) {
        this.resources = resources;
    }

    public synchronized void down() {
        while (resources < 1) {
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {

                e.printStackTrace();
            }
        }
        resources--;
    }

    public synchronized void up() {
        resources++;
    }

}