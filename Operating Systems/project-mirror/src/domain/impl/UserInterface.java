package domain.impl;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import domain.api.ControlInterface;
import domain.api.NotificationInterface;
import domain.api.SubmissionInterface;

/**
 * This class is the thread responsible for all the interactions
 * with the user.
 */
public class UserInterface extends Thread implements NotificationInterface {
    private ControlInterface controlInterface;
    private SubmissionInterface submissionInterface;

    private JFrame frame = new JFrame("Process Scheduling Simulator");
    private JPanel basePanel = new JPanel();
    private JPanel buttonsPanel = new JPanel();
    private JPanel jobPanel = new JPanel();
    private JButton startButton = new JButton("Start");
    private JButton resumeButton = new JButton("Resume");
    private JButton suspendButton = new JButton("Suspend");
    private JButton stopButton = new JButton("Stop");
    private JButton showQueuesButton = new JButton("Show Queues");
    private JButton submitJobButton = new JButton("Submit Job");
    private JButton clearButton = new JButton("Clear output");
    private JTextField jobTextField = new JTextField(20);
    private JTextArea reportTextArea = new JTextArea(34, 30);
    private JScrollPane reportScrollPane = new JScrollPane(reportTextArea);


    /**
     * Creates a window for the simulator instance.
     *
     * @param title - window title
     */
    public UserInterface(String title) {
        frame.setTitle(title);
        startButton.setEnabled(true);
        resumeButton.setEnabled(false);
        suspendButton.setEnabled(false);
        stopButton.setEnabled(false);

        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controlInterface.startSimulation();

                startButton.setEnabled(false);
                suspendButton.setEnabled(true);
                stopButton.setEnabled(true);
            }
        });

        resumeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controlInterface.resumeSimulation();

                suspendButton.setEnabled(true);
                resumeButton.setEnabled(false);
            }
        });

        suspendButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controlInterface.suspendSimulation();

                suspendButton.setEnabled(false);
                resumeButton.setEnabled(true);
            }
        });

        stopButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controlInterface.stopSimulation();

                startButton.setEnabled(true);
                suspendButton.setEnabled(false);
                resumeButton.setEnabled(false);
                stopButton.setEnabled(false);
            }
        });

        showQueuesButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                submissionInterface.displaySubmissionQueue();
                controlInterface.displayProcessQueues();
            }
        });

        submitJobButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                submitJob();
            }
        });

        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                reportTextArea.setText("");;
            }
        });

        jobTextField.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                submitJob();
            }
        });

        reportTextArea.setEditable(false);
        reportTextArea.setLineWrap(true);
        reportTextArea.setWrapStyleWord(true);
        reportScrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);

        buttonsPanel.setLayout(new BoxLayout(buttonsPanel, BoxLayout.X_AXIS));
        buttonsPanel.add(startButton);
        buttonsPanel.add(resumeButton);
        buttonsPanel.add(suspendButton);
        buttonsPanel.add(stopButton);
        buttonsPanel.add(showQueuesButton);

        jobPanel.setLayout(new BoxLayout(jobPanel, BoxLayout.X_AXIS));
        jobPanel.add(jobTextField);
        jobPanel.add(submitJobButton);
        jobPanel.add(clearButton);

        basePanel.add(buttonsPanel);
        basePanel.add(jobPanel);
        basePanel.add(reportScrollPane);

        basePanel.setPreferredSize(new Dimension(455, 580));
        basePanel.setLayout(new BoxLayout(basePanel, BoxLayout.Y_AXIS));

        frame.add(basePanel);

        frame.setResizable(false);

        frame.pack();
        frame.setLayout(null);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    /**
     * Sets the controlInterface instance to be used.
     *
     * @param controlInterface - instance of `ControlInterface`
     */
    public void setControlInterface(ControlInterface controlInterface) {
        this.controlInterface = controlInterface;
    }

    /**
     * Sets the submissionInterface instance to be used.
     * 
     * @param submissionInterface - instance of `SubmissionInterface`
     */
    public void setSubmissionInterface(SubmissionInterface submissionInterface) {
        this.submissionInterface = submissionInterface;
    }

    /**
     * Appends information to the output section of the window.
     *
     * @param info - information to be appended to the output section of the
     * window.
     */
    @Override
    public void display(String info) {
        reportTextArea.append("\n------------------------------------\n\n" + info + "\n");
        reportTextArea.setCaretPosition(reportTextArea.getDocument().getLength());
    }

    /**
     * Helper method to reduce code repetition when submiting a job (used in
     * action listeners to submit a job whenever the user clicks the submit job
     * button or presses enter in the job text field).
     */
    private void submitJob() {
        String input = jobTextField.getText();
        if (input.equals("")) {
            return;
        }
        var submitted = submissionInterface.submitJob(input);
        if (submitted) {
            jobTextField.setText("");
        }
    }

}
