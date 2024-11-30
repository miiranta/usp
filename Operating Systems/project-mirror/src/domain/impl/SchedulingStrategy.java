package domain.impl;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Queue;

import domain.api.NotificationInterface;

/**
 * Abstract class to declare methods that all Strategies must have, and reuse
 * common code.
 */
public abstract class SchedulingStrategy {
    public Queue<domain.impl.Process> ready;
    public Queue<domain.impl.Process> blocked;
    public Queue<domain.impl.Process> finished = new ArrayDeque<>();

    protected int quantumSizeMs;
    protected NotificationInterface notificationInterface;
    protected domain.impl.Process executing;

    /**
     * Setter for `notificationInterface`.
     *
     * @param notificationInterface - instance of `NotificationInterface`
     */
    public void setNotificationInterface(NotificationInterface notificationInterface) {
        this.notificationInterface = notificationInterface;
    }

    /**
     * Executes a line of code through a quantum of time.
     */
    public abstract void execute();

    /**
     * Whether or not a process is currently executing.
     *
     * @return boolean - whether or not a process is currently executing.
     */
    public boolean isExecutingProcess() {
        return executing != null;
    }

    /**
     * Sets the currently executing process to `null`.
     */
    public void resetExecuting() {
        executing = null;
    }

    /**
     * Setter for the quantum size
     * 
     * @param sizeMs int - size of the quantum in ms.
     */
    public void setQuantumSizeMs(int sizeMs) {
        quantumSizeMs = sizeMs;
    }

    /**
     * Method to simulate the passing of a quantum of time. It will decrement
     * the blocked time of all processes in the blocked queue and add them to
     * the ready queue if they are finished.
     *
     * @throws InterruptedException - if the `Thread` is interrupted
     */
    protected void passQuantum() throws InterruptedException {
        Thread.sleep(quantumSizeMs);
        decrementBlockedTimes();
    }

    /**
     * Decrement the blocked time of all processes in the blocked queue,
     * placing the ones that achieved blocked time of zero in the ready queue.
     */
    private void decrementBlockedTimes() {
        var toRemove = new ArrayList<>();

        for (var blockedProcess : blocked) {
            blockedProcess.decrementBlockedTime();

            if (blockedProcess.getBlockedFor() == 0) {
                if (blockedProcess.hasNextLine()) {
                    ready.add(blockedProcess);
                } else {
                    finished.add(blockedProcess);
                }
                toRemove.add(blockedProcess);
            }
        }

        blocked.removeAll(toRemove);
    }
}
