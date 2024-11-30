package domain.impl;

import java.util.ArrayDeque;
import java.util.NoSuchElementException;

/**
 * This scheduling strategy uses ArrayDeque for its ready queue, so that
 * whenever it executes a process, it removes it from the front of ready queue,
 * executes it for a quantum of time, then puts it at the end of the ready or
 * blocked queue, according to the last instruction.
 */
public class RoundRobin extends SchedulingStrategy {

    /**
     * Instantiates a new `RoundRobin` that extends `SchedulingStrategy` with
     * ready and blocked queues as new `ArrayDeque`s.
     */
    public RoundRobin() {
        ready = new ArrayDeque<>();
        blocked = new ArrayDeque<>();
    }

    /**
     * Gets the first process in the ready queue and simulates the execution of
     * one instruction of the process, displaying the process beign executed,
     * passing the time of a quantum, and then adding it in the ready, blocked
     * or finished queue, depending on the instruction read. If no process gets
     * executed, just passes one quantum.
     */
    @Override
    public void execute() {
        try {
            executing = ready.remove();

            if (executing.hasNextLine()) {
                var line = executing.getNextLine();

                try {
                    notificationInterface.display("RoundRobin: \nExecuting: " + executing.getName());
                    passQuantum();

                    // This means the simulation was stopped by another thread
                    if (executing == null) {
                        return;
                    }

                    if (line.getBlockFor() > 0) {
                        executing.setBlockedFor(line.getBlockFor());
                        blocked.add(executing);
                    } else if (executing.hasNextLine()) {
                        ready.add(executing);
                    } else {
                        finished.add(executing);
                    }
                } catch (InterruptedException ie) { }
            } else {
                finished.add(executing);
            }
        } catch (NoSuchElementException e) {
            try {
                passQuantum();
            } catch (InterruptedException ie) { }
        } finally {
            executing = null;
        }
    }

}
