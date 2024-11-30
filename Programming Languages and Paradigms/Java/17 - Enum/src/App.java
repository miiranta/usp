public class App {
    
    //Enum is for enumerating constants
    enum iKnowHowtoCount{
        ONE,
        TWO,
        THREE,
        FOUR;
    }
    //CONSTANTS in CAPS!
    
    public static void main(String[] args) throws Exception {
        
        
        //Declaring var
        iKnowHowtoCount enum1 = iKnowHowtoCount.TWO;
        System.out.println(enum1);


        //Using Switch
        iKnowHowtoCount switch1 = iKnowHowtoCount.ONE;

        switch (switch1){
        case ONE:
            System.out.println("ONE ;o.");
            break;
        case TWO:
            System.out.println("TWO ;o ;O.");
            break;
        default:
            System.out.println("Idk.");
            break;
        }


        //Printing all values
        for (iKnowHowtoCount loop : iKnowHowtoCount.values()) {
            System.out.println(loop + " YAAAAy");
        }


    }
}
