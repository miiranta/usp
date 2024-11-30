package domain.impl;

import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;

import domain.api.ControlInterface;
import domain.api.InterSchedulerInterface;
import domain.api.NotificationInterface;

/**
 * This class is responsible for executing a scheduling strategy in a given list
 * of processes that it receives by implementing the `InterSchedulerInterface`.
 * It also implements `ControlInterface` so that the simulation can be
 * manipulated.
 */
public class ShortTermScheduler extends Thread implements ControlInterface, InterSchedulerInterface {
    private NotificationInterface notificationInterface;
    private SchedulingStrategy schedulingStrategy;

    private boolean isRunning = false;

    // This is important to avoid ConcurrentModificationException, because
    // while the queue is being printed, the LongTermScheduler can add a
    // process to the ready queue
    private Semaphore readySemaphore = new Semaphore(1);

    /**
     * Constructor that receives a quantum size in milliseconds and a scheduling
     * strategy so that the simulation can be executed with given params.
     *
     * @param schedulingStrategy - scheduling strategy to be used
     * @param quantumSizeMs - quantum size in milliseconds
     */
    public ShortTermScheduler(SchedulingStrategy schedulingStrategy, int quantumSizeMs) {
        this.schedulingStrategy = schedulingStrategy;
        this.schedulingStrategy.setQuantumSizeMs(quantumSizeMs);
    }

    /**
     * Setter for the `notificationInterface`.
     *
     * @param notificationInterface - instance of `NotificationInterface`
     */
    public void setNotificationInterface(NotificationInterface notificationInterface) {
        this.notificationInterface = notificationInterface;
        this.schedulingStrategy.setNotificationInterface(notificationInterface);
    }

    /**
     * Used to add a process to the ready queue.
     *
     * @param process - Process instance that will be added to the ready queue
     */
    @Override
    public void addProcess(Process process) {
        try {
            readySemaphore.acquire();
            schedulingStrategy.ready.add(process);
            readySemaphore.release();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Returns the total number of processes currently managed by this
     * scheduler. Is equal to the number of processes in the ready queue, plus
     * the number of processes in the blocked queue, plus one, if it is
     * currently executing a process.
     */
    @Override
    public int getProcessLoad() {
        return schedulingStrategy.ready.size() + schedulingStrategy.blocked.size()
                + (schedulingStrategy.isExecutingProcess() ? 1 : 0);
    }

    /**
     * Starts the simulation, calling the `start()` method of this thread if it
     * wasn't alive and notify the user of the event.
     */
    @Override
    public void startSimulation() {
        isRunning = true;

        if (!isAlive()) {
            start();
        }

        notificationInterface.display("ShortTermScheduler: \nSimulation started!");
    }

    /**
     * Pauses the simulation and notify the user of the event.
     */
    @Override
    public void suspendSimulation() {
        isRunning = false;

        notificationInterface.display("ShortTermScheduler: \nSimulation suspended!");
    }

    /**
     * Resumes the simulation and notify the user of the event.
     */
    @Override
    public void resumeSimulation() {
        isRunning = true;

        notificationInterface.display("ShortTermScheduler: \nSimulation resumed!");
    }

    /**
     * Stops the simulation, clearing all the queues and notify the user of the
     * event.
     */
    @Override
    public void stopSimulation() {
        isRunning = false;
        restart();

        notificationInterface
                .display("ShortTermScheduler: \nSimulation stopped! \nAll processes were removed from the queues.");
    }

    /**
     * Displays the names of the processes in all the queues.
     */
    @Override
    public void displayProcessQueues() {
        String processQueues;
        var readyQueue = "\n> Ready:\t";
        var blockedQueue = "\n> Blocked:\t";
        var finishedQueue = "\n> Finished:\t";
        List<String> readyNames;

        try {
            readySemaphore.acquire();
            readyNames = schedulingStrategy.ready.stream().map(p -> p.getName()).collect(Collectors.toList());
            readySemaphore.release();
        } catch (InterruptedException e) {
            e.printStackTrace();
            return;
        }

        readyQueue += String.join(", ", readyNames);
        var blockedNames = schedulingStrategy.blocked.stream().map(p -> p.getName()).collect(Collectors.toList());
        blockedQueue += String.join(", ", blockedNames);
        var finishedNames = schedulingStrategy.finished.stream().map(p -> p.getName()).collect(Collectors.toList());
        finishedQueue += String.join(", ", finishedNames);

        processQueues = String.join("\n", readyQueue, blockedQueue, finishedQueue);
        notificationInterface.display("ShortTermScheduler: \nProcess queues \n" + processQueues);
    }

    /**
     * When the `Thread` is started, this method will run an infinite loop,
     * executing the `schedulingStrategy` until the simulation is stopped. If
     * the simulation is stopped or suspended, it will sleep for 1 ms each
     * iteration.
     */
    @Override
    public void run() {
        while (true) {

            // This is important to avoid the CPU to be 100% used DONT REMOVE IT
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
                return;
            }

            if (!isRunning) {
                continue;
            }

            if (getProcessLoad() > 0) {
                schedulingStrategy.execute();
                displayProcessQueues();
            }
        }
    }

    /**
     * Clears all queues.
     */
    private void restart() {
        schedulingStrategy.ready.clear();
        schedulingStrategy.blocked.clear();
        schedulingStrategy.finished.clear();
        schedulingStrategy.resetExecuting();

        return;
    }

}
