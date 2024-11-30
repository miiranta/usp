package application;

import domain.impl.LongTermScheduler;
import domain.impl.SchedulingStrategy;
import domain.impl.ShortTermScheduler;
import domain.impl.UserInterface;

/**
 * This class is responsible for creating all objects necessary for a
 * simulation, with the correct parameters, and creating the relationships
 * between them.
 */
public class SchedulerSimulator {
    private UserInterface userInterface;
    private LongTermScheduler longTermScheduler;
    private ShortTermScheduler shortTermScheduler;

    /**
     * This method is responsible for creating an instance of the simulator,
     * given quantum size, max load and strategy.
     *
     * @param maxShortTermSchedulerLoad - max load of `ShortTermScheduler`
     * @param quantumSizeMs - quantum size in milliseconds
     * @param schedulingStrategy - scheduling strategy to use in this simulation
     */
    public SchedulerSimulator(int maxShortTermSchedulerLoad, int quantumSizeMs, SchedulingStrategy schedulingStrategy) {
        shortTermScheduler = new ShortTermScheduler(schedulingStrategy, quantumSizeMs);
        longTermScheduler = new LongTermScheduler(maxShortTermSchedulerLoad);
        userInterface = new UserInterface(schedulingStrategy.getClass().getSimpleName() + ", max load: "
                + maxShortTermSchedulerLoad + ", quantum: " + quantumSizeMs + "ms");

        shortTermScheduler.setNotificationInterface(userInterface);
        longTermScheduler.setInterSchedulerInterface(shortTermScheduler);
        longTermScheduler.setNotificationInterface(userInterface);
        userInterface.setControlInterface(shortTermScheduler);
        userInterface.setSubmissionInterface(longTermScheduler);

        longTermScheduler.start();
        userInterface.start();
    }
}
