import java.text.DecimalFormat;
import java.util.Arrays;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import application.SchedulerSimulator;
import domain.impl.RoundRobin;
import domain.impl.SchedulingStrategy;
import domain.impl.ShortestJobFirst;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.text.DefaultFormatterFactory;
import javax.swing.text.NumberFormatter;

public class App {
    static JFrame frame = new JFrame("Process Scheduling Simulator");
    static JPanel mainPanel = new JPanel();
    static JLabel strategyLabel = new JLabel("Scheduling strategy:", SwingConstants.CENTER);
    static JComboBox<String> strategyComboBox = new JComboBox<>(new String[] { "Shortest Job First", "Round Robin" });
    static JLabel loadLabel = new JLabel("Max ShortTermScheduler load (int >0):", SwingConstants.CENTER);
    static JFormattedTextField loadField = new JFormattedTextField();
    static JLabel quantumLabel = new JLabel("Quantum size in milliseconds (int >0):", SwingConstants.CENTER);
    static JFormattedTextField quantumField = new JFormattedTextField();
    static JPanel buttonsPanel = new JPanel();
    static JButton startButton = new JButton("Start");
    static JButton exitButton = new JButton("Exit");

    public static void main(String[] args) throws Exception {
        var argsList = Arrays.asList(args);
        int npIndex = argsList.indexOf("-np");
        int quantumIndex = argsList.indexOf("-qt");
        int stratIndex = argsList.indexOf("-strat");
        int maxShortTermSchedulerLoad = 10;
        int quantumSizeMs = 200;
        String stratName = "sjf";
        SchedulingStrategy strategy;

        if (npIndex != -1) { // the -np flag was used
            try {
                maxShortTermSchedulerLoad = Integer.parseInt(args[npIndex + 1]);
                if (maxShortTermSchedulerLoad <= 0) {
                    throw new NumberFormatException();
                }
            } catch (ArrayIndexOutOfBoundsException | NumberFormatException e) {

                // if the user uses the flag and don't inform a value subsequently
                if (e instanceof ArrayIndexOutOfBoundsException) {
                    throw new ArrayIndexOutOfBoundsException("If using -np flag, inform a value (>0)");
                }

                // if the value informed isn't valid (couldn't convert to int, or is <= 0)
                if (e instanceof NumberFormatException) {
                    throw new NumberFormatException("Inform a valid value for np (>0)");
                }

                throw e;
            }
        }

        if (quantumIndex != -1) {
            try {
                quantumSizeMs = Integer.parseInt(args[quantumIndex + 1]);
                if (quantumSizeMs <= 0) {
                    throw new NumberFormatException();
                }
            } catch (Exception e) {
                if (e instanceof NumberFormatException) {
                    throw new NumberFormatException("Inform a valid value for qt (>0)");
                }
                if (e instanceof ArrayIndexOutOfBoundsException) {
                    throw new ArrayIndexOutOfBoundsException("If using -qt flag, inform a value (>0)");
                }
            }
        }

        if (stratIndex != -1) { // the -strat flag was used
            try {
                stratName = args[stratIndex + 1];
                if (!stratName.equals("sjf") && !stratName.equals("rr")) {
                    throw new IllegalArgumentException();
                }
            } catch (ArrayIndexOutOfBoundsException | NumberFormatException e) {

                // if the user uses the flag and don't inform a value subsequently
                if (e instanceof ArrayIndexOutOfBoundsException) {
                    throw new ArrayIndexOutOfBoundsException("If using -strat flag, inform a value ('rr' or 'sjf')");
                }

                // if the value informed wasn't valid (not 'sjf' nor 'rr')
                if (e instanceof IllegalArgumentException) {
                    throw new IllegalArgumentException("Inform a valid strategy ('rr' or 'sjf')");
                }

                throw e;
            }
        }

        if (stratName.equals("rr")) {
            strategy = new RoundRobin();
        } else {
            strategy = new ShortestJobFirst();
        }

        createGUI();

        if (npIndex != -1 || stratIndex != -1 || quantumIndex != -1) {
            new SchedulerSimulator(maxShortTermSchedulerLoad, quantumSizeMs, strategy);
        }

    }

    /**
     * Creates a window that allows the user to configure and start simulations.
     */
    static void createGUI() {
        startButton.addActionListener(e -> {
            SchedulingStrategy strategy = null;
            int maxLoad = 0;
            int quantumSizeMs = 0;

            if (strategyComboBox.getSelectedItem().equals("Shortest Job First")) {
                strategy = new ShortestJobFirst();
            } else if (strategyComboBox.getSelectedItem().equals("Round Robin")) {
                strategy = new RoundRobin();
            } else {
                JOptionPane.showMessageDialog(frame, "Invalid strategy", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            try {
                maxLoad = Integer.parseInt(loadField.getText());
                if (maxLoad <= 0) {
                    throw new NumberFormatException();
                }
            } catch (NumberFormatException err) {
                JOptionPane.showMessageDialog(frame, "Invalid load", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            try {
                quantumSizeMs = Integer.parseInt(quantumField.getText());
                if (quantumSizeMs <= 0) {
                    throw new NumberFormatException();
                }
            } catch (NumberFormatException err) {
                JOptionPane.showMessageDialog(frame, "Invalid quantum size", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            if (strategy != null) {
                new SchedulerSimulator(maxLoad, quantumSizeMs, strategy);
            }
        });

        exitButton.addActionListener(e -> {
            frame.dispose();
            System.exit(0);
        });

        var decimalFormat = new DecimalFormat("#,##0");
        decimalFormat.setGroupingUsed(false);
        var numberFormatter = new NumberFormatter(decimalFormat);

        loadField.setFormatterFactory(new DefaultFormatterFactory(numberFormatter));
        quantumField.setFormatterFactory(new DefaultFormatterFactory(numberFormatter));

        loadField.setValue(10);
        quantumField.setValue(200);

        mainPanel.add(strategyLabel);
        mainPanel.add(strategyComboBox);
        mainPanel.add(quantumLabel);
        mainPanel.add(quantumField);
        mainPanel.add(loadLabel);
        mainPanel.add(loadField);
        mainPanel.add(buttonsPanel);
        buttonsPanel.add(startButton);
        buttonsPanel.add(exitButton);

        loadField.setColumns(5);

        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        frame.add(mainPanel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
