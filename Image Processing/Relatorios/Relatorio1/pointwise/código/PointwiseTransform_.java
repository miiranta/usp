
import java.awt.*;
import java.awt.event.*;
import ij.*;
import ij.gui.*;
import ij.process.*;

public class Inverse_ {

	public Inverse_() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp.getType() != ImagePlus.GRAY8) {
			IJ.showMessage("Only process the 8-bit image");
			return;
		}
		ImageAccess input = new ImageAccess(imp.getProcessor());
		ImageAccess output = PointwiseTransform.inverse(input);
		(new ImagePlus("Inverse of " + imp.getTitle(), output.createByteProcessor())).show();
	}
}

public class Rescale_ {

	public Rescale_() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp.getType() != ImagePlus.GRAY8 &&
			imp.getType() != ImagePlus.GRAY16 &&
			imp.getType() != ImagePlus.GRAY32
			) {
			IJ.showMessage("Only process the 8-bit, 16-bit or 32-bit image");
			return;
		}
		ImageAccess input = new ImageAccess(imp.getProcessor());
		ImageAccess output = PointwiseTransform.rescale(input);
		(new ImagePlus("Rescale of " + imp.getTitle(), output.createByteProcessor())).show();
	}
}

public class Saturate_ {

	public Saturate_() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp.getType() != ImagePlus.GRAY8 &&
			imp.getType() != ImagePlus.GRAY16 &&
			imp.getType() != ImagePlus.GRAY32
			) {
			IJ.showMessage("Only process the 8-bit, 16-bit or 32-bit image");
			return;
		}
		ImageAccess input = new ImageAccess(imp.getProcessor());
		ImageAccess output = PointwiseTransform.saturate(input);
		output.show("Saturate of " + imp.getTitle());
	}
}

/**
public class SelectColor_ {

	public SelectColor_() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp.getType() != ImagePlus.COLOR_RGB) {
			IJ.showMessage("Only process the color RGB image");
			return;
		}
		GenericDialog gd = new GenericDialog("Select a Color");
		gd.addNumericField("Red ", 128, 0);
		gd.addNumericField("Green ", 128, 0);
		gd.addNumericField("Blue ", 128, 0);
		gd.addNumericField("Tolerance ", 10, 0);
		gd.showDialog();
		if (gd.wasCanceled())
			return;
			
		ColorProcessor cp = (ColorProcessor)imp.getProcessor();
		int nx = imp.getWidth();
		int ny = imp.getHeight();
		byte[] r = new byte[nx*ny];
		byte[] g = new byte[nx*ny];
		byte[] b = new byte[nx*ny];
		cp.getRGB(r, g, b);
		ByteProcessor ipr = new ByteProcessor(nx, ny);
		ipr.setPixels(r);
			
		ByteProcessor ipg = new ByteProcessor(nx, ny);
		ipg.setPixels(g);
		
		ByteProcessor ipb = new ByteProcessor(nx, ny);
		ipb.setPixels(b);
		
		ImageAccess output = PointwiseTransform.selectColor(
				new ImageAccess(ipr), 
				new ImageAccess(ipg), 
				new ImageAccess(ipb), 
				gd.getNextNumber(),
				gd.getNextNumber(),
				gd.getNextNumber(),
				gd.getNextNumber());
		(new ImagePlus("Color selection of " + imp.getTitle(), output.createByteProcessor())).show();
	}
	
}
*/

public class ZMIP_ {

	public ZMIP_() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp.getType() != ImagePlus.GRAY8 &&
			imp.getType() != ImagePlus.GRAY16 &&
			imp.getType() != ImagePlus.GRAY32
			) {
			IJ.showMessage("Only process the 8-bit, 16-bit or 32-bit image");
			return;
		}
		int nz = imp.getStack().getSize();
		ImageAccess[] input = new ImageAccess[nz];
		for (int z = 1; z <= nz; z++) {
			input[z-1] = new ImageAccess(imp.getStack().getProcessor(z));
		}
		ImageAccess output = PointwiseTransform.zprojectMaximum(input);
		(new ImagePlus("Z MIP of " + imp.getTitle(), output.createByteProcessor())).show();
	}
}

public class ZMean_ {

	public ZMean_() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp.getType() != ImagePlus.GRAY8 &&
			imp.getType() != ImagePlus.GRAY16 &&
			imp.getType() != ImagePlus.GRAY32
			) {
			IJ.showMessage("Only process the 8-bit, 16-bit or 32-bit image");
			return;
		}
		int nz = imp.getStack().getSize();
		ImageAccess[] input = new ImageAccess[nz];
		for (int z = 1; z <= nz; z++) {
			input[z-1] = new ImageAccess(imp.getStack().getProcessor(z));
		}
		ImageAccess output = PointwiseTransform.zprojectMean(input);
		(new ImagePlus("Z Mean of " + imp.getTitle(), output.createByteProcessor())).show();
	}
}

