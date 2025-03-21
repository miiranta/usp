import java.awt.*;
import javax.swing.*;
import ij.*;
import ij.gui.*;
import ij.process.*;

/**
 * <p><b>Plugin of ImageJ:</b><br>
 * Edge Detector Session<br>
 *
 */	

public class Measure_Level {

	/**
	* Constructor creates the dialog box.
	*/
	public Measure_Level() {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp == null) {
		    IJ.error("Input image required.");
		    return;
		}
		int nx = imp.getWidth();
		int ny = imp.getHeight();
		int nt = imp.getStack().getSize();
		
		LCanvas canvas = new LCanvas(imp);
		imp.setWindow(new StackWindow(imp, canvas));
		
		DisplayTools.setImagePlus(imp, canvas);
		
		ImageAccess[] sequence = new ImageAccess[nt];
		for(int t=0; t<nt; t++) {
			sequence[t] = new ImageAccess(imp.getStack().getProcessor(t+1));
			IJ.showStatus("Reading images ...");
			imp.setSlice(t+1);
		}
		CodeClass.measureLevel(sequence);
		imp.setSlice(1);		
	}

	//
	public class LCanvas extends ImageCanvas {

		private ImagePlus imp;
		private int line[];
		private int level1[];
		private int level2[];
		private int nx;
		private int ny;
		private int nt;
		
		/**
		*/
		public LCanvas(ImagePlus imp) {
	    	super(imp);
			this.imp = imp;
			nx = imp.getWidth();
			ny = imp.getHeight();
			nt = imp.getStack().getSize();
			line = new int[nt];
			level1 = new int[nt];
			level2 = new int[nt];
			for(int t=0; t<nt; t++) {
				line[t] = -1;
				level1[t] = -1;
			}
		}

		/**
		*/
		public void setLine(int t, int y) {
			if (t >= 0 && t <nt)
				line[t] = y;
			repaint();
		}

		/**
		*/
		public void setLevels(int t, int y1, int y2) {
			if (t >= 0 && t <nt) {
				level1[t] = y1;
				level2[t] = y2;
			}
			repaint();
		}
		
		/**
		*/
		public void paint(Graphics g) {
			super.paint(g);
			double mag = getMagnification();
			Rectangle rect = getSrcRect();
			Color saveColor = g.getColor();
			
			int t = imp.getCurrentSlice()-1;
			long x1 = Math.round((0-rect.x+0.5)*mag);
			long x2 = Math.round((nx-rect.x+0.5)*mag);
			if (level1[t] > 0) {
				long y1 = Math.round((Math.min(level1[t],level2[t])-rect.y+0.5)*mag);
				long y2 = Math.round((Math.abs(level2[t]-level1[t])-rect.y+0.5)*mag);
				g.setColor(new Color(0, 255, 128, 128));
				g.fillRect((int)x1, (int)y1, (int)x2, (int)y2);
			}
			if (line[t] > 0) {	
				long y = Math.round((line[t]-rect.y+0.5)*mag);
				g.setColor(new Color(255, 0, 0));
				g.drawLine((int)x1, (int)y, (int)x2, (int)y);
			}
		}
	}
}

