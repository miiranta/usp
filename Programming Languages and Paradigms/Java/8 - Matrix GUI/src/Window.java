import javax.swing.*;

import java.awt.*;
import java.util.Arrays;

public class Window {
    
    public static double matrix[][];
    private static JFrame frame;
    private static JPanel panel;
    private static JPanel panel1; 
    private static JPanel panel2;  
    private static JPanel panel3; 
    private static JButton button;
    private static JTextArea textArea;
    private static JScrollPane scrollPane;
    private static WindowListener list = new WindowListener();
    
    Window(){
        createWindow();
        addWindowPanel();
        addWindowPanelLayout();
    }

    public static void createWindow(){
        frame = new JFrame("Matrix Calculator - By miranda :3");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridBagLayout());
    }

    public static void addWindowPanel(){
        panel = new JPanel();
        panel.setBorder(BorderFactory.createLineBorder(Color.black));

        panel1 = new JPanel();
        panel1.setLayout(new GridLayout(2, 1));
        panel1.setPreferredSize(new Dimension(300, 300));
        panel1.setBorder(BorderFactory.createLineBorder(Color.black));

        panel2 = new JPanel();
        panel2.setBorder(BorderFactory.createEmptyBorder(0, 50, 0, 50));

        panel3 = new JPanel();
        panel3.setLayout(new GridLayout(6, 1));
        panel3.setPreferredSize(new Dimension(300, 300));
        panel3.setBorder(BorderFactory.createLineBorder(Color.black));

        JButton multiplyByK = new JButton("Multiply Line By K");
        JButton multiplyByKAndSumLine = new JButton("Multiply Line By K and Sum to another");
        JButton swapLine = new JButton("Swap Lines");

        //Panel multiplyByKListener
        JPanel p1 = new JPanel();
        p1.setLayout(new GridLayout(1, 4));
        p1.add(new JLabel("K:"));
        JTextField k1 = new JTextField(3);
        p1.add(k1);
        p1.add(new JLabel("Line:"));
        JTextField l1 = new JTextField(3);
        p1.add(l1);

        //Panel multiplyByKAndSumLine
        JPanel p2 = new JPanel();
        p2.setLayout(new GridLayout(1, 6));
        p2.add(new JLabel("K:"));
        JTextField k2 = new JTextField(3);
        p2.add(k2);
        p2.add(new JLabel("Line 1:"));
        JTextField l2 = new JTextField(3);
        p2.add(l2);
        p2.add(new JLabel("Line 2:"));
        JTextField l3 = new JTextField(3);
        p2.add(l3);

        //Panel swapLine
        JPanel p3 = new JPanel();
        p3.setLayout(new GridLayout(1, 4));
        p3.add(new JLabel("Line 1:"));
        JTextField l4 = new JTextField(3);
        p3.add(l4);
        p3.add(new JLabel("Line 2:"));
        JTextField l5 = new JTextField(3);
        p3.add(l5);

        panel3.add(multiplyByK);
        panel3.add(p1);
        panel3.add(multiplyByKAndSumLine);
        panel3.add(p2);
        panel3.add(swapLine);
        panel3.add(p3);

        list.multiplyByKListener(multiplyByK, k1, l1);
        list.multiplyByKAndSumLineListener(multiplyByKAndSumLine, k2, l2, l3);
        list.swapLineListener(swapLine, l4, l5);
    }

    public static void addWindowPanelLayout(){
        button = new JButton("Add matrix");
        textArea = new JTextArea(5, 10);
        scrollPane = new JScrollPane(textArea); 

        textArea.setEditable(true);

        panel1.add(scrollPane);
        panel1.add(button);
        panel.add(panel1);
        frame.add(panel);

        list.addListener(button, textArea);
    }

    public static void setWindowVisible(){
        frame.pack();
        frame.setResizable(true);
        frame.setVisible(true);
    }

    public static void addMatrix(double[][] matrixLocal){
        panel2.removeAll();
        matrix = matrixLocal;

        panel2.setLayout(new GridLayout(matrix.length, matrix.length));
        for(int i = 0; i<matrix.length; i++){
            panel2.add(new JLabel(Arrays.toString(matrix[i]).replace(",", "").replace("[", "").replace("]", "").trim()))
                  .setFont(new Font("AvantGarde", Font.BOLD, 20));
        }

        panel.add(panel2);
        panel.add(panel3);
        frame.pack();
    }

}
