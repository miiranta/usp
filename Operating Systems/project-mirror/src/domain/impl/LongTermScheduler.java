package domain.impl;

import domain.api.InterSchedulerInterface;
import domain.api.NotificationInterface;
import domain.api.SubmissionInterface;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.nio.file.NoSuchFileException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * This class is responsible for getting a path to a file and creating a
 * `Process` instance with its contents, then submitting this new object to
 * the `ShortTermScheduler` via an instance of `NotificationInterface`.
 */
public class LongTermScheduler extends Thread implements SubmissionInterface {
    private InterSchedulerInterface interSchedulerInterface = null;
    private NotificationInterface notificationInterface = null;

    private List<Process> submissionQueue = new ArrayList<Process>();

    private final int MAX_SUBMISSION_QUEUE_SIZE = 10000;
    private final int MAX_SHORT_TERM_SCHEDULER_LOAD;
    private final int SLEEP_FOR_MS = 100;

    /**
     * Constructor for LongTermScheduler, accepts the max `ShortTermScheduler`
     * load as argument, to balance the system's load.
     * @param maxShortTermSchedulerLoad - max load for `ShortTermScheduler`
     */
    public LongTermScheduler(int maxShortTermSchedulerLoad) {
        this.MAX_SHORT_TERM_SCHEDULER_LOAD = maxShortTermSchedulerLoad;
    }

    /**
     * Setter for `IntrSchedulerInterface` instance.
     * @param interSchedulerInterface - instance of `InterSchedulerInterface`
     */
    public void setInterSchedulerInterface(InterSchedulerInterface interSchedulerInterface) {
        this.interSchedulerInterface = interSchedulerInterface;
    }

    /**
     * Setter for `NotificationInterface` instance.
     * @param notificationInterface - instance of `NotificationInterface`
     */
    public void setNotificationInterface(NotificationInterface notificationInterface) {
        this.notificationInterface = notificationInterface;
    }

    /**
     * Method used to create a `Process` instance from a file path, then
     * submitting it to the `ShortTermScheduler` via an instance of
     * `NotificationInterface`. Can also submit all files inside a directory.
     *
     * @param fileName - path to file or directory
     * @return boolean - true if file was submitted successfully, false otherwise
     */
    @Override
    public boolean submitJob(String fileName) {
        var fileOrDir = new File(fileName);

        if (!fileOrDir.exists()) {
            notificationInterface.display("LongTermScheduler: \nError! File not found.");
            return false;
        }

        // if file is a directory, then recursively call submitJob for each file
        // in directory
        if (fileOrDir.isDirectory()) {
            var files = fileOrDir.listFiles();

            for (var file : files) {
                if (file.isFile()) {
                    var newFilePath = Paths.get(fileName, file.getName()).toString();
                    submitJob(newFilePath);
                }
            }

            return true;
        }

        if (submissionQueue.size() >= MAX_SUBMISSION_QUEUE_SIZE) {
            notificationInterface.display("LongTermScheduler: \nFull submission queue.");
            return false;
        }

        try {
            var newProcess = new Process(fileName);

            submissionQueue.add(newProcess);
            displaySubmissionQueue();

            return true;

        } catch (IOException | IllegalArgumentException e) {
            if (e instanceof NoSuchFileException) {
                notificationInterface.display("LongTermScheduler: \nError! File not found.");
            } else if (e instanceof IllegalArgumentException) {
                notificationInterface.display("LongTermScheduler: \nError! " + e.getMessage());
            } else {
                e.printStackTrace();
                notificationInterface.display("LongTermScheduler: \nError! See terminal for more details.");
            }
            return false;

        }
    }

    /**
     * Displays all the processes that are in the submission queue of this
     * scheduler waiting to be submitted to the ShortTermScheduler.
     */
    @Override
    public void displaySubmissionQueue() {
        String output = "LongTermScheduler:\n> Submission Queue: ";

        var processes = submissionQueue.stream()
                .map(p -> p.getName())
                .collect(Collectors.toList());

        output += String.join(", ", processes);
        notificationInterface.display(output);

        return;
    }

    /**
     * When the `Thread` is started, this method will continuously check if
     * there are processes to be submitted, and submit them if the
     * `ShortTermScheduler`'s load is below some limit.
     */
    @Override
    public void run() {
        while (true) {

            if (submissionQueue.size() > 0) {

                if (interSchedulerInterface.getProcessLoad() < MAX_SHORT_TERM_SCHEDULER_LOAD) {
                    var newProcess = submissionQueue.remove(0);

                    interSchedulerInterface.addProcess(newProcess);

                    notificationInterface.display("LongTermScheduler: \nProcess " + newProcess.getName() + " added to the ready queue.");
                }

            }

            try {
                Thread.sleep(SLEEP_FOR_MS);
            } catch (InterruptedException e) {
                notificationInterface.display("LongTermScheduler: \nLongTermScheduler interrupted.");
            }

        }

    }

}
