import ij.process.ImageProcessor;
import ij.process.ByteProcessor;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;

public class Programa {
    public static void main(String[] args) {

        try {
            
        
            InverterS inverter = new InverterS();

            // Create a new image processor - amogus.jpg
            BufferedImage img = ImageIO.read(new File("amogus.png"));
            ImageProcessor ip = new ByteProcessor(img);

            // Run the inverter plugin
            inverter.run(ip);

            // Save the image
            ImageIO.write(ip.getBufferedImage(), "png", new File("amogus_inverted.png"));
        
        }

        catch (Exception e) {
            e.printStackTrace();
        }
        

    }
}