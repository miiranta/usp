package domain.api;

/**
 * Interface that allows the simulation to be controlled.
 */
public interface ControlInterface {

    /**
     * Starts the simulation.
     */
    public void startSimulation();

    /**
     * Suspends (pauses) the simulation.
     */
    public void suspendSimulation();

    /**
     * Resumes a suspended simulation.
     */
    public void resumeSimulation();

    /**
     * Completely stops the simulation.
     */
    public void stopSimulation();

    /**
     * Displays the current state of all queues.
     */
    public void displayProcessQueues();
}
