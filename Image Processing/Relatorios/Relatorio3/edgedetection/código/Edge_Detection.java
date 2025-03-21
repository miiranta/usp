import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import ij.*;
import ij.gui.*;
import ij.process.*;

/**
 * <p><b>Plugin of ImageJ:</b><br>
 * Edge Detector Session<br>
 *
 */	

public class Edge_Detection extends JDialog implements ActionListener, ItemListener {

	private ImagePlus impSource;			// Reference to the input image

	private static double sigma = 3.0;		
	private static double thresholdLow = 10.0;		
	private static double thresholdHigh = 50.0;
	private static double conversionPixMM  = 10.0;

	/**
	* Constructor creates the dialog box.
	*/
	public Edge_Detection() {
		super(new Frame(), "Edge Detection");
		if (IJ.versionLessThan("1.20a"))
			return;
		doDialog();
	}

	/**
	* Check the image.
	*/
	private boolean checkImage() {
		if (impSource == null) {
		    IJ.error("Input image required.");
		    return false;
		}

		if (impSource.getType() == ImagePlus.COLOR_256 || impSource.getType() == ImagePlus.COLOR_RGB) {
			IJ.error("Do not process the color images.");
			return false;
		}
		
		int nx = impSource.getWidth();
		int ny = impSource.getHeight();

		if (nx <= 3) {
			IJ.error("The input image should greater than 3.");
			return false;
		}

		if (ny <= 3) {
			IJ.error("The input image should greater than 3.");
			return false;
		}
		
		return true;
	}

	/**
	* Check the parameters.
	*/
	private boolean checkParameters()
	{
		if (sigma < 0.001) 
			sigma = 0.001;
		return true;
	}

	/**
	* Build the dialog box.
	*/
	private GridBagLayout 		layout				= new GridBagLayout();;
	private GridBagConstraints 	constraint			= new GridBagConstraints();;
	private JLabel 				lblSigma			= new JLabel("Sigma", Label.RIGHT);;
	private JTextField			txtSigma			= new JTextField("3.0", 5);
	private JLabel 				lblThresholdLow		= new JLabel("Low in %", Label.RIGHT);;;
	private JLabel 				lblThresholdHigh	= new JLabel("High in %", Label.RIGHT);;
	private JTextField 			txtThresholdLow		= new JTextField("10", 5);;
	private JTextField 			txtThresholdHigh	= new JTextField("50", 5);;
	private JButton 			bnRun				= new JButton("Run Student Solution");;
	private JButton 			bnRunTeacher		= new JButton("Run Teacher Solution");;
	private JButton 			bnClose				= new JButton("Close");;
	private JCheckBox 			chkSmoothing 		= new JCheckBox("Gaussian blurring", true);
	private JCheckBox 			chkGradient	 		= new JCheckBox("Gradient computation, module, gx and gy", true);
	private JCheckBox 			chkSuppression		= new JCheckBox("Non-Maximum Suppression", true);;
	private JCheckBox 			chkThreshold		= new JCheckBox("Threshold", true);;

	private void doDialog() {
					
		// Panel buttons 
		JPanel pnButtons = new JPanel();
		pnButtons.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 0));
		
		pnButtons.add(bnRunTeacher);
		pnButtons.add(bnClose);
		pnButtons.add(bnRun);

		// Panel main 
		JPanel pnMain = new JPanel();
		pnMain.setLayout(layout);
		pnMain.setBorder(BorderFactory.createEtchedBorder());
	 	addComponent(pnMain, 0, 0, 1, 1, 5, chkSmoothing);
		addComponent(pnMain, 0, 1, 1, 1, 5, lblSigma);
		addComponent(pnMain, 0, 2, 1, 1, 5, txtSigma);
		addComponent(pnMain, 1, 0, 5, 1, 5, chkGradient);
		addComponent(pnMain, 2, 0, 5, 1, 5, chkSuppression);
		addComponent(pnMain, 3, 0, 1, 1, 5, chkThreshold);
		addComponent(pnMain, 3, 1, 1, 1, 5, lblThresholdLow);
		addComponent(pnMain, 3, 2, 1, 1, 5, txtThresholdLow);
		addComponent(pnMain, 3, 3, 1, 1, 5, lblThresholdHigh);
		addComponent(pnMain, 3, 4, 1, 1, 5, txtThresholdHigh);
		

		// Add Listeners
		bnClose.addActionListener(this);
		bnRun.addActionListener(this);
		bnRunTeacher.addActionListener(this);
		chkSmoothing.addItemListener(this);
		chkGradient.addItemListener(this);
		chkSuppression.addItemListener(this);
		chkThreshold.addItemListener(this);

		JPanel pnMain1 = new JPanel();
		pnMain1.setLayout(layout);
		addComponent(pnMain1, 0, 0, 1, 1, 10, pnMain);
		addComponent(pnMain1, 1, 0, 1, 1, 10, pnButtons);
		
		// Building the main panel
		this.getContentPane().add(pnMain1);
		pack();
		setResizable(false);
		GUI.center(this);
		setVisible(true);
		IJ.wait(250); 	// work around for Sun/WinNT bug
	}

	/**
	* Add a component in a panel in the northeast of the cell.
	*/
	private void addComponent(JPanel pn, int row, int col, int width, int height, int space, JComponent comp) {
	    constraint.gridx = col;
	    constraint.gridy = row;
	    constraint.gridwidth = width;
	    constraint.gridheight = height;
	    constraint.anchor = GridBagConstraints.NORTHWEST;
	    constraint.insets = new Insets(space, space, space, space);
		constraint.weightx = IJ.isMacintosh()?90:100;
		constraint.fill = constraint.HORIZONTAL;
	    layout.setConstraints(comp, constraint);
	    pn.add(comp);
	}

	/**
	* Implements the actionPerformed for the ActionListener.
	*/
	public synchronized  void actionPerformed(ActionEvent e) {
		if (e.getSource() == bnClose) {
			
			dispose();
		}

		else if (e.getSource() == bnRun) {
			sigma = getDoubleValue(txtSigma, Double.MIN_VALUE, 3.0, Double.MAX_VALUE);
			thresholdLow  = getDoubleValue(txtThresholdLow, Double.MIN_VALUE, 20.0, Double.MAX_VALUE);
			thresholdHigh = getDoubleValue(txtThresholdHigh, Double.MIN_VALUE, 60.0, Double.MAX_VALUE);
			if (sigma < 0.001) 
				sigma = 0.001;
			run(false);
			
		}

		else if (e.getSource() == bnRunTeacher) {
			sigma = getDoubleValue(txtSigma, Double.MIN_VALUE, 3.0, Double.MAX_VALUE);
			thresholdLow  = getDoubleValue(txtThresholdLow, Double.MIN_VALUE, 20.0, Double.MAX_VALUE);
			thresholdHigh = getDoubleValue(txtThresholdHigh, Double.MIN_VALUE, 60.0, Double.MAX_VALUE);
			if (sigma < 0.001) 
				sigma = 0.001;
			run(true);
			
		}
	}

	/**
	* Implements the itemStateChanged for the ItemListener.
	*/
	public synchronized void itemStateChanged(ItemEvent e) {

		if (e.getSource() == chkSmoothing) {
			if (chkSmoothing.isSelected()) {
				lblSigma.show();
				txtSigma.show();
			}
			else {
				lblSigma.hide();
				txtSigma.hide();
			}
		}
		
		else if (e.getSource() == chkGradient) {
			if (chkGradient.isSelected()) {
			}
			else {
				chkSuppression.setSelected(false);
				chkThreshold.setSelected(false);
			}
			enableThreshold();
		}
		
		else if (e.getSource() == chkSuppression) {
			if (chkSuppression.isSelected()) {
				chkGradient.setSelected(true);
			}
			else {
				chkThreshold.setSelected(false);
			}
			enableThreshold();
		}

		else if  (e.getSource() == chkThreshold) {
			if (chkThreshold.isSelected()) {
				chkGradient.setSelected(true);
				chkSuppression.setSelected(true);
			}
			enableThreshold();
		}

		notify();
	}

	/**
	*/
	private void enableThreshold() {
		if (chkThreshold.isSelected()) {
			txtThresholdLow.show();
			lblThresholdLow.show();
			txtThresholdHigh.show();
			lblThresholdHigh.show();
		}
		else {
			txtThresholdHigh.hide();
			lblThresholdHigh.hide();
			txtThresholdLow.hide();
			lblThresholdLow.hide();
		}
	}

	/**
	* Get a double value from a TextField between minimal and maximal values.
	*/
	private double getDoubleValue(JTextField text, double mini, double defaut, double maxi) {
		double d;
		try {
			d = (new Double(text.getText())).doubleValue();
			if (d < mini)  
				text.setText( "" + mini);
			if (d > maxi)  
				text.setText( "" + maxi);
		}
		
		catch (Exception e) {
			if (e instanceof NumberFormatException) 
				text.setText( "" + defaut);
		}
		d = (new Double(text.getText())).doubleValue();
		return d;
	}

	/**
	* Process the image
	*/
	private void run(boolean teacher) {
		impSource = WindowManager.getCurrentImage();
		
		if (checkImage() == false)
			return;
		
		int nx = impSource.getWidth();
		int ny = impSource.getHeight();
		int nz = impSource.getStack().getSize();
		
		ImageAccess smooth = null;
		ImageAccess grad[] = null;
		ImageAccess thinEdge = null;
		ImageAccess threshold = null;
		
		ImageStack stack = new ImageStack(nx, ny);
		String title = "";
		for(int z=0; z<nz; z++) {
	 		ImageAccess image = new ImageAccess(impSource.getStack().getProcessor(z+1));
	 		impSource.setSlice(z+1);
	 		IJ.showStatus("" + (z+1) + "/" + nz);
			if (chkSmoothing.isSelected())
				smooth = (teacher ? TeacherCode.blurring(image, sigma) : CodeClass.blurring(image, sigma));
			else
				smooth = image.duplicate();
			
			if (chkGradient.isSelected())
				grad = (teacher ? TeacherCode.gradient(smooth) : CodeClass.gradient(smooth));
			
			if (chkSuppression.isSelected())
				thinEdge = (teacher ? TeacherCode.suppressNonMaximum(grad) : CodeClass.suppressNonMaximum(grad));
					
			if (chkThreshold.isSelected())
				threshold = TeacherCode.doHysteresisThreshold(thinEdge, thresholdLow, thresholdHigh);
					
			if (!chkGradient.isSelected())
			if (!chkSuppression.isSelected())
			if (!chkThreshold.isSelected()) {
				stack.addSlice("", smooth.createFloatProcessor());
				title = "Blurring";
			}
			
			if (chkGradient.isSelected())
			if (!chkSuppression.isSelected())
			if (!chkThreshold.isSelected()) {
				stack.addSlice("Module", grad[0].createFloatProcessor());
				stack.addSlice("Gx", grad[1].createFloatProcessor());
				stack.addSlice("Gy", grad[2].createFloatProcessor());
				title = "Gradient";
			}
			
			if (chkSuppression.isSelected())
			if (!chkThreshold.isSelected()) {
				stack.addSlice("", thinEdge.createFloatProcessor());
				title = "Non-Maximum Suppression";
			}

			if (chkThreshold.isSelected()) {
				ByteProcessor bp = threshold.createByteProcessor();
				bp.invert();
				stack.addSlice("", bp);
				title = "Threshold";
			}
		}
		(new ImagePlus(title, stack)).show();
	}
}


