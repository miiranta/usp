public class Constructor {
    
    Constructor(){
        //Call itself
        this("calling the other constructor");
    }

    //Overloading of the constructor
    Constructor(String a){
        System.out.print(a);
    }

}
