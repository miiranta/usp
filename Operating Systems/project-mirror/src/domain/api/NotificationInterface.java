package domain.api;

/**
 * Interface used to display notifications to the user.
 */
public interface NotificationInterface {
    /**
     * Notify any information of interest related to the simulation to the
     * UserInterface by the schedulers.
     * 
     * @param info - Information to be notified.
     */
    void display(String info);
}
