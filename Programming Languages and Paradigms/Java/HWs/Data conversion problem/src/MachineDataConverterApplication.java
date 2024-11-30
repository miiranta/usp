/** 
*   @author Lucas Miranda 12542838
*
*
*/

import java.util.Scanner;

import model.MachineData;
import adapters.MachineDataConverter;
import streams.MachineDataConverterInputStream;
import streams.MachineDataConverterOutputStream;

public class MachineDataConverterApplication {

    private static Scanner applicationInput = new Scanner(System.in);

    private static void showMenu() {
        System.out.println("[1] Insert file name");
        System.out.println("[0] Exit application");
    }

    private static int chooseOption() {
        showMenu();

        int option;
        option = applicationInput.nextInt();
        applicationInput.nextLine();
        return option;
    }

    private static void runSolution(String applicationStream) {
        MachineDataConverterInputStream inputStream = new MachineDataConverterInputStream(applicationStream);

        MachineData machineData = inputStream.streamDataToMachineData();

        MachineDataConverter machineDataConverter = new MachineDataConverter();

        MachineData convertedMachineData = machineDataConverter.convertMachineDataToNewModel(machineData);

        MachineDataConverterOutputStream outputStream = new MachineDataConverterOutputStream(applicationStream);

        outputStream.contentToFile(convertedMachineData.toString());
    }

    public static void main(String[] args) {

        while(chooseOption() != 0) {

            System.out.println("Insira o nome do arquivo: ");
            String fileName = applicationInput.nextLine();

            runSolution(fileName);
            
        } 
    }
}
