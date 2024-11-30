// Classe que implementa interface de conversão para o modelo verde (GREEN)

package adapters.converting;

public class ConvertingStrategyFromModelGreen implements ConvertingStrategy{
    
    public String convertContent(String contentToConvert){
        String convertedContent = "";
        String[] splittedContentToConvert = contentToConvert.split(" ");
        for(String elem : splittedContentToConvert) {
            String asciiRepresentaion = parseBinaryToChar(elem);
            convertedContent += asciiRepresentaion;
        }
        return convertedContent;
    }

    public String parseBinaryToChar(String elem){

        int a = Integer.parseInt(elem, 2);
        char resChar = (char)(a);
        return String.valueOf(resChar);

    }


}
