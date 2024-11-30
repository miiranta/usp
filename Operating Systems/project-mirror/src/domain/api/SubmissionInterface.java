package domain.api;

/**
 * Interface used to standardize submissions of new processes.
 */
public interface SubmissionInterface {
    
    /**
     * Submits a new process by path.
     *
     * @param fileName - directory path of the process or folder containing the
     * processes
     * @return `true` if the submission was successful, `false` otherwise
     */
    boolean submitJob(String fileName);

    /**
     * Shows the processes waiting to get to the ready queue.
     */
    void displaySubmissionQueue();
}
