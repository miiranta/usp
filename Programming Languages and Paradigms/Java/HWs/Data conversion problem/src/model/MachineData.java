package model;

/**
 * Representation of a machine data model
 * every machine data has a model as well as the information, or content, itself
 */

public class MachineData {
    /**
     * Represents the MachineData model
     */
    private MachineDataModel machineDataModel;
    private String machineContent;

    /**
     * Default behaviour is to construct a MachineData that follows the new model
     * @param content
     */
    public MachineData (String content) {
        this(content, MachineDataModel.NEW);
    }

    /**
     * Constructs MachineData given its content and model
     * @param content
     * @param model
     */
    public MachineData(String content, MachineDataModel model) {
        setMachineContent(content);
        setMachineDataModel(model);
    }

    /**
     * Constructs MachineData given its content and model code
     * the model code will be used to identify the corresponding model in the enum
     * @param content
     * @param modelCode
     */
    public MachineData (String content, int modelCode) {
        this(content, identifyMachineDataModel(modelCode));
    }

    /**
     * Setter for the machineDataModel object attribute
     * @param model
     */
    private void setMachineDataModel(MachineDataModel model) {
        machineDataModel = model;
    }

    /**
     * Setter for the machineContent object attribute
     * @param content
     */
    private void setMachineContent(String content) {
        machineContent = content;
    }

    /**
     *
     * @return the content disregarding metadata
     */
    public String getMachineContent() {
        return machineContent;
    }

    /**
     * 
     * @return the enum value of the data model
     */
    public MachineDataModel getMachineDataModel() {
        return machineDataModel;
    }

    /**
     * 
     * @param modelCode
     * @return data model enum value based on a given code
     */
    private static MachineDataModel identifyMachineDataModel(int modelCode) {
        if(modelCode == 1) {
            return MachineDataModel.BLUE;
        }
        else if (modelCode == 2) {
            return MachineDataModel.GREEN;
        }
        else if (modelCode == 3) {
            return MachineDataModel.YELLOW;
        }
        else if (modelCode == 4) {
            return MachineDataModel.NEW;
        }
        else {
            return MachineDataModel.UNKNOWN;
        }
    }

    /**
     * 
     * @return stringfied object with both model and content
     */
    @Override
    public String toString() {
        return machineDataModel.toString() + "\n" + machineContent; 
    }

}
