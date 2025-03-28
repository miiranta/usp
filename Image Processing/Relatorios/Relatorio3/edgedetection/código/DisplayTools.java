import java.awt.*;
import ij.*;
import ij.gui.*;
import ij.process.*;
import ij.text.*;

public class DisplayTools {

	private static ImagePlus imp;
	private static Measure_Level.LCanvas canvas;

	/**
	*/
	public static void setImagePlus(ImagePlus imp1, Measure_Level.LCanvas canvas1) {
		imp = imp1;
		canvas = canvas1;
	}

	/**
	*/
	public static void plot(float[] func, String labelAbscissa, String labelOrdinate) {
		if (func==null)
			return;
		int n = func.length;
		float x[] = new float[n];
        for(int k=0; k<n; k++) {
        	x[k] = k;
        }
        PlotWindow plot = new PlotWindow("Plot", labelAbscissa, labelOrdinate, x, func);
        plot.draw();
	}
	
	/**
	*/
	public static void plot(double[] func, String labelAbscissa, String labelOrdinate) {
		if (func==null)
			return;
		int n = func.length;
		float funcf[] = new float[n];
		for(int k=0; k<n; k++) {
        	funcf[k] = (float)func[k];
        }
        plot(funcf, labelAbscissa, labelOrdinate);
	}
	
	/**
	*/
	public static void plot(int[] func, String labelAbscissa, String labelOrdinate) {
		if (func==null)
			return;
		int n = func.length;
		float funcf[] = new float[n];
		for(int k=0; k<n; k++) {
        	funcf[k] = (float)func[k];
        }
        plot(funcf, labelAbscissa, labelOrdinate);
	}
	
	/**
	*/
	public static void drawLine(int t, int y) {
		imp.setSlice(t+1);
		canvas.setLine(t, y);
		canvas.repaint();
	}
	
	/**
	*/
	public static void drawLevels(int t, int y1, int y2) {
		imp.setSlice(t+1);
		canvas.setLevels(t, y1, y2);
		canvas.repaint();
	}

}

