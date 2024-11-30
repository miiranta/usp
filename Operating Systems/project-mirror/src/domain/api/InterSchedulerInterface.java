package domain.api;

import domain.impl.Process;

/**
 * Allows for communication between schedulers.
 */
public interface InterSchedulerInterface {

    /**
     * Add a process to the `ShortTermScheduler`'s ready queue.
     *
     * @param process - process to be added to the `ShortTermScheduler`'s ready queue.
     */
    void addProcess(Process process);

    /**
     * Returns the current load of the `ShortTermScheduler`.
     *
     * @return quantity of processes currently managed by `ShortTermScheduler`
     */
    int getProcessLoad();
}
