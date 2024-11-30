import One.onesayhello;
// Package.Class

public class App {

    //(ALL FUNCTIONS ARE STATIC, so they dont need to be instanced in an object!)

    public static void main(String[] args) throws Exception {
        
       
            //When it's in the SAME FOLDER, it's implicitly imported
            yo.sayYo(); //Package "yo"

            //When the folder is different, you can do 2 things

                //IMPORT and use normally (recommended)
                onesayhello.sayHello();

                //Use the FULL DIRECTORY
                Two.twosaybye.sayBye();

    }
}
