package adapters;

import adapters.converting.*;
import adapters.converting.ConvertingStrategy;
import model.MachineData;
import model.MachineDataModel;

/**
 * A MachineDataConverter converts a given MachineData's content to new model
 * it does so by using a converting strategy
 */
public class MachineDataConverter {
    private ConvertingStrategy convertingStrategy;

    /**
     * Given a MachineData object, attempts to return another with content in new model
     * if it is not possible to define a converting strategy, returns the original object
     * @param machineData
     * @return a MachineData in new model
     */
    public MachineData convertMachineDataToNewModel(MachineData machineData) {
        boolean couldDefineStrategy = defineConvertingStrategyByModel(machineData.getMachineDataModel());
        
        if(couldDefineStrategy) {
            String newContent = convertContentByDefinedStrategy(machineData.getMachineContent());
            return new MachineData(newContent);
        }
        
        return machineData;
    }

    /**
     * Defines which strategy is going to be used based on model
     * @param model
     * @return boolean stating if it was possible to determine a strategy
     */
    private boolean defineConvertingStrategyByModel(MachineDataModel model) {
        
	/**
     * model.toString() retorna o modelo do computador dado pelo código
     * Podemos compará-lo para decidir o que será a variavel convertingStrategy
     * a variavel convertingStrategy pode ser definida com uma instância baseada na interface ConvertingStrategy dependendo do modelo escolhido
	 */

        if(model.toString() == "1"){
            //Há um método implementado para cada modelo, atribuimos convertingStrategy para uma nova instancia que permita fazer a conversão desejada
            convertingStrategy = new ConvertingStrategyFromModelBlue();
            return true;
        }
        if(model.toString() == "2"){
            convertingStrategy = new ConvertingStrategyFromModelGreen();
            return true;
        }
        if(model.toString() == "3"){
            convertingStrategy = new ConvertingStrategyFromModelYellow();
            return true;
        }

        //Se nenhum modelo é compativel com os definidos acima, retorna false e faz o código prosseguir sem conversão
        System.out.print("There is no strategy for this model!\n");
        return false;

    }

    /**
     * Delegate content convertion to ConvertingStrategy abstraction
     * @param content
     * @return a string containing the converted content
     */
    private String convertContentByDefinedStrategy(String content) {
        String convertedContent = convertingStrategy.convertContent(content);
        return convertedContent;
    }
}
