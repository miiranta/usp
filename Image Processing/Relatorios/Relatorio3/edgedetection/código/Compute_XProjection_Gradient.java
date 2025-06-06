import ij.*;
import ij.process.*;

public class Compute_XProjection_Gradient {

	/**
	*/
	public Compute_XProjection_Gradient() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp == null) {
		    IJ.error("Input image required.");
		    return;
		}
		if (imp.getStack().getSize() != 1) {
		    IJ.error("Single image required.");
		    return;
		}
		
		ImageAccess image = new ImageAccess(imp.getProcessor());
		Code.computeXProjectionGradient(image);
	}
}
