package domain.impl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.regex.Pattern;

/**
 * Class that represents a process that will be executed by the simulation.
 */
public class Process {
    private String name;
    private int priority;
    private List<ProgramLine> body = new ArrayList<>();
    private String rawContent;
    private int timeEstimate;
    private int lineNumber = 0;
    private int blockedFor = 0;

    /**
     * Constructor that takes a file path as a parameter to create an instance
     * of class `Process`.
     *
     * @param filePath - path to file containing program lines
     * @throws IOException - if there's and error reading the file
     */
    public Process(String filePath) throws IOException {
        try {
            rawContent = Files.readString(Paths.get(filePath), StandardCharsets.UTF_8).replaceAll("\r", "");

            setHeader();
            setBody();
            makeTimeEstimate();
        } catch (IOException e) {
            throw e;
        }
    }

    /**
     * Getter for this process' `name` property.
     *
     * @return name of this process
     */
    public String getName() {
        return name;
    }

    /**
     * Getter for this process' `blockedFor` property.
     *
     * @return amount of quanta that this process will be blocked for
     */
    public int getBlockedFor() {
        return blockedFor;
    }

    /**
     * Setter for this process' `blockedFor` property.
     *
     * @param blockedFor amount of quanta that this process will be blocked for
     */
    public void setBlockedFor(int blockedFor) {
        this.blockedFor = blockedFor;
    }

    /**
     * Returns the next line that will be executed.
     *
     * @return next line that will be executed
     * @throws NoSuchElementException if there are no more lines to execute
     */
    public ProgramLine getNextLine() throws NoSuchElementException {
        if (hasNextLine()) {
            return body.get(lineNumber++);
        }

        throw new NoSuchElementException();
    }

    /**
     * Returns `true` if there are more lines to execute, `false` otherwise.
     *
     * @return `true` if there are more lines to execute, `false` otherwise
     */
    public boolean hasNextLine() {
        return lineNumber < body.size();
    }

    /**
     * Getter for this process' `timeEstimate` property.
     *
     * @return an estimate for the amount of time quanta this process will take
     * to execute
     */
    public int getTimeEstimate() {
        return timeEstimate;
    }

    /**
     * Decrements this process' blockedTime.
     */
    public void decrementBlockedTime() {
        if (blockedFor > 0) {
            blockedFor--;
        }
    }

    /**
     * Parses program header from raw content.
     *
     * @throws IllegalArgumentException
     */
    private void setHeader() throws IllegalArgumentException {
        var headerPattern = Pattern.compile("^program ([a-zA-Z0-9.]+) (\\d)$", Pattern.MULTILINE);
        var headerMatcher = headerPattern.matcher(rawContent);

        if (headerMatcher.find()) {
            name = headerMatcher.group(1);
            priority = Integer.parseInt(headerMatcher.group(2));
        } else {
            throw new IllegalArgumentException("Invalid program header. Raw input:\n" + rawContent);
        }
    }

    /**
     * Parses program body from raw content.
     *
     * @throws IllegalArgumentException
     */
    private void setBody() throws IllegalArgumentException {
        var contentPattern = Pattern.compile("begin\n([a-zA-Z0-5 \n]+)end\n?");
        var contentMatcher = contentPattern.matcher(rawContent);

        if (contentMatcher.find()) {
            var lines = contentMatcher.group(1).split("\n");

            for (var line : lines) {
                var programLine = new ProgramLine(line);
                body.add(programLine);
            }
        } else {
            throw new IllegalArgumentException("Invalid program content. Raw input:\n" + rawContent);
        }
    }

    /**
     * Estimates total process execution time by summing time estimates of
     * individual program lines.
     */
    private void makeTimeEstimate() {
        for (var line : body) {
            timeEstimate += line.getTimeEstimate();
        }
    }

    /**
     * Returns a string representation of this process.
     */
    @Override
    public String toString() {
        return super.toString() + "\n" +
                "Program name: " + name + "\n" +
                "Priority: " + priority + "\n" +
                "Time estimate: " + timeEstimate + "\n" +
                "Body:\n" + body + "\n" +
                "Raw content:\n" + rawContent + "\n";
    }
}
