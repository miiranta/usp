package utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Class with utility methods not belonging to the domain of this project.
 */
public class Utils {


    /**
     * Returns the index of the first element in the list that is greater than
     * or equal to the target. Would be used in conjunction with a `List` so
     * that we could insert an element into the list according to some property,
     * in our case, a `Process`' `timeEstimate`, to use with the Shortest Job
     * First (SJF) algorithm. By using a `PriorityQueue`, this became obsolete.
     *
     * @param list - list of elements on which to find the upper bound
     * @param target - target value for upper bound
     * @param getter - function to get the property with which to compare
     * elements.
     */
    public static <T, S extends Comparable<S>> int upperBound(List<T> list, S target, Function<T, S> getter) {
        int l = 0;
        int r = list.size() - 1;
        
        while (l < r) {
            int m = (l + r) / 2;
            var element = list.get(m);
            var value = getter.apply(element);

            if (value.compareTo(target) <= 0) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }

        return r;
    }

    /**
     * Generates a random set of programs with filenames 1.txt, 2.txt, ...,
     * n.txt, in the simulated programs language specified in the assignment.
     *
     * @param n - number of programs to generate
     */
    public static void generateRandomPrograms(int n) {
        for (int i : IntStream.range(1, n + 1).toArray()) {
            var filename = i + ".txt";
            var process = Utils.makeProgram(filename, 20);

            try {
                var writer = new BufferedWriter(new FileWriter("data/processes/" + filename));
                writer.write(process);
                writer.close();
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Makes one process at a time.
     *
     * @param filename - name of the program and file
     * @param maxBodySize - maximum number of lines in the process body
     */
    private static String makeProgram(String filename, int maxBodySize) {
        int priority = ThreadLocalRandom.current().nextInt(0, 9);
        var program = "program " + filename + " " + priority + "\nbegin\n";

        int nLines = ThreadLocalRandom.current().nextInt(1, maxBodySize);

        for (var line : IntStream.range(0, nLines).toArray())  {
            if (ThreadLocalRandom.current().nextBoolean()) {
                program += "execute\n";
            } else {
                var waitFor = ThreadLocalRandom.current().nextInt(1, 6);
                program += "block " + waitFor + "\n";
            }
        }

        program += "end\n";

        return program;
    }
}
