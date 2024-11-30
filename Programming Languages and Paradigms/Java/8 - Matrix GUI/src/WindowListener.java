import java.awt.event.*; 
import javax.swing.*;

public class WindowListener{

    public void addListener(JButton button, JTextArea textArea){

        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                
                if("".equals(textArea.getText())){
                    System.out.println("The space is empty!");
                }else{
                    double readyMatrix[][] = Parser.parseMatrix(textArea.getText());
                    if(readyMatrix != null){

                        //PRINTS MATRIX!
                        System.out.println();
                        for (int i = 0; i < readyMatrix.length; i++) {
                            for (int j = 0; j < readyMatrix[i].length; j++) {
                                System.out.print(readyMatrix[i][j] + " ");
                            }
                            System.out.println();
                        }
                        System.out.println();

                        Window.addMatrix(readyMatrix);

                    }
                }
            
            }
        });

    }

    public void multiplyByKListener(JButton button, JTextField k, JTextField line1){

        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                
                double matrix[][] = Window.matrix;
                int l1 = Integer.parseInt(line1.getText());
                double k1 = Double.parseDouble(k.getText());

                if(l1 > matrix.length || l1 == 0){
                    System.out.println("Line not set!");
                    return;
                }

                Parser.multiplyByK(l1, k1, matrix);

            }
        });

    }

    public void multiplyByKAndSumLineListener(JButton button, JTextField k, JTextField line1, JTextField line2){

        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                
                double matrix[][] = Window.matrix;
                int l1 = Integer.parseInt(line1.getText());
                int l2 = Integer.parseInt(line2.getText());
                double k1 = Double.parseDouble(k.getText());

                if(l1 > matrix.length || l2 > matrix.length || l1 == 0 || l2 == 0){
                    System.out.println("Line not set!");
                    return;
                }

                Parser.multiplyByKAndSumLine(l1, l2, k1, matrix);
            
            }
        });

    }

    public void swapLineListener(JButton button, JTextField line1, JTextField line2){

        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                
                double matrix[][] = Window.matrix;
                int l1 = Integer.parseInt(line1.getText());
                int l2 = Integer.parseInt(line2.getText());
        
                if(l1 > matrix.length || l2 > matrix.length || l1 == 0 || l2 == 0){
                    System.out.println("Line not set!");
                    return;
                }
            
                Parser.swapLine(l1, l2, matrix);

            }
        });

    }

}
