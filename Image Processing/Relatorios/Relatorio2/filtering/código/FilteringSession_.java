/*
 * Filtering session.
 *
 * 
 *
 */

import ij.*;
import ij.text.*;
import ij.gui.*;
import ij.process.*;
import java.awt.*;
import java.awt.event.*;


public class FilteringSession_ extends Dialog
	implements ActionListener, ItemListener, WindowListener	{
	
	private ImagePlus imp;			// Reference to the input image

	private List 				lstOperation 	= new List(9);
	private Button 				bnRunSession 	= new Button("Run Student Solution");;
	private Button 				bnRunSolution 	= new Button("Run Teacher Solution");;
	private Button 				bnClose 		= new Button("Close");;
	private Label 				lblLength 		= new Label("Length of the window");
	private TextField 			txtLength 		= new TextField("5", 10);
	private GridBagLayout 		layout			= new GridBagLayout();;
	private GridBagConstraints 	constraint		= new GridBagConstraints();
	private TextWindow			journal;	
	/**
	* Constructor.
	*/
	public FilteringSession_() {
		super(new Frame(), "Digital Filtering");
		
		if (IJ.versionLessThan("1.21a"))
			return;
			
		lstOperation.add("Vertical Edge Non-Separable");
		lstOperation.add("Vertical Edge Separable");
		lstOperation.add("Horizontal Edge Non-Separable");
		lstOperation.add("Horizontal Edge Separable");
		lstOperation.add("Moving-Average 5*5 Non-Separable");
		lstOperation.add("Moving-Average 5*5 Separable");
		lstOperation.add("Moving-Average 5*5 Recursive");
		lstOperation.add("Moving-Average L*L Recursive");
		lstOperation.add("Sobel Edge Detector");
		lstOperation.select(0);
		
		Panel pnParameters = new Panel();
		pnParameters.setLayout(layout);
	 	addComponent(pnParameters, 0, 0, 2, 1, 5, lstOperation);
	 	addComponent(pnParameters, 1, 0, 1, 1, 5, lblLength);
	 	addComponent(pnParameters, 1, 1, 1, 1, 5, txtLength);
	 	addComponent(pnParameters, 2, 0, 1, 1, 5, bnClose);
	 	addComponent(pnParameters, 2, 1, 1, 1, 5, bnRunSolution);
	 	addComponent(pnParameters, 3, 1, 1, 1, 5, bnRunSession);
		
		lstOperation.addItemListener(this);
		bnRunSession.addActionListener(this);
		bnRunSolution.addActionListener(this);
		bnClose.addActionListener(this);	 
		
 		addWindowListener(this);
		add(pnParameters);
		pack();
		GUI.center(this);
		setVisible(true);
		
	}

	/**
	*/
	public void runSession() {
	
		// Input image
		imp = WindowManager.getCurrentImage();
		if (imp == null) {
			IJ.error("Input image required");
			return;
		}
		
	 	if (checkImage() == false)
	 		return;
		
		// Parameters
		int nx = imp.getWidth();
		int ny = imp.getHeight();
		String operation = lstOperation.getSelectedItem();
	    int length = getIntegerValue( txtLength, 1, 3, 1000);
		if ( ((length-1)/2)*2 != (length-1)) {
			IJ.showMessage("The length should be odd.");
			return;
		}
		if ( length < 1) {
			IJ.showMessage("The length should greater than 1.");
			return;
		}
		if ( length >= nx || length >= ny) {
			IJ.showMessage("The length should smaller than the size.");
			return;
		}
		
		ImageStack inputstack=imp.getStack();
		int slices = inputstack.getSize();
		ImageStack stack = new ImageStack(nx, ny);
		ImageAccess out = null;
		
		// Processing
		double startTime = System.currentTimeMillis();
		String title = "[Student] ";
		String impl = "";
		String filter = "";

		for (int plane=1; plane <= slices; plane++) {
		
			ImageAccess image  = new ImageAccess (inputstack.getProcessor(plane)); 
			
			if (operation.equals("Vertical Edge Non-Separable")) {
				out = FilteringSession.detectEdgeVertical_NonSeparable(image);
				impl = "non-sep";
				filter = "Vertical Edge";
				title += "Verti, Non-Sep";
			}  
			else if (operation.equals("Vertical Edge Separable")) {
				out = FilteringSession.detectEdgeVertical_Separable(image);
				impl = "sep";
				filter = "Vertical Edge";
				title += "Verti, Sep";
			}  
			else if (operation.equals("Horizontal Edge Non-Separable")) {
				out = FilteringSession.detectEdgeHorizontal_NonSeparable(image);
				impl = "non-sep";
				filter = "Horizontal Edge";
				title += "Horiz, Non-Sep";
			}  
			else if (operation.equals("Horizontal Edge Separable")) {
				out = FilteringSession.detectEdgeHorizontal_Separable(image);
				impl = "sep";
				filter = "Horizontal Edge";
				title += "Horiz, Sep";
			}  
			else if (operation.equals("Moving-Average 5*5 Non-Separable")) {
				out = FilteringSession.doMovingAverage5_NonSeparable(image);
				impl = "non-sep";
				filter = "Moving-Average 5*5";
				title += "MA5x5, Non-Sep";
			}  
			else if (operation.equals("Moving-Average 5*5 Separable")) {
				out = FilteringSession.doMovingAverage5_Separable(image);
				impl = "sep";
				filter = "Moving-Average 5*5";
				title += "MA5x5, Sep";
			}  
			else if (operation.equals("Moving-Average 5*5 Recursive")) {
				out = FilteringSession.doMovingAverage5_Recursive(image);
				title += "MA5x5, Recursive";
				filter = "Moving-Average 5*5";
				impl = "recursive";
			}  
			else if (operation.equals("Moving-Average L*L Recursive")) {
				out = FilteringSession.doMovingAverageL_Recursive(image, length);
				title += "MALxL, Recursive";
				impl = "recursive";
				filter = "Moving-Average " + length + "*" + length;
			}  
			else if (operation.equals("Sobel Edge Detector")) {
				out = FilteringSession.doSobel(image);
				title += "Sobel";
				impl = "";
				filter = "Sobel";
			}
			stack.addSlice("", out.createFloatProcessor());
		}

		if (journal != null) {
			if (!journal.isVisible())
				journal = null;	
		}
		if (journal == null) {
			journal = new TextWindow("Journal", "Input Image\tFilter\tImpl.\tTime [ms]\tOutput Image\tMean(out)\tMin(out)\tMax(out)", "", 700, 250);
			Point loc = this.getLocation();
			Dimension dim = this.getSize();
			Point locj = journal.getLocation();
			journal.setLocation(locj.x, loc.y+dim.height+10);
			journal.show();
		}

		ImagePlus impResult = new ImagePlus(title, stack);
		impResult.show();

		String time = IJ.d2s(System.currentTimeMillis() - startTime);
		String maxi = IJ.d2s(out.getMaximum());
		String mini = IJ.d2s(out.getMinimum());
		String mean = IJ.d2s(out.getMean());
		journal.append("" + imp.getTitle() + "\t" + filter + "\t" + impl + "\t" + time + "\t" + impResult.getTitle() + "\t" + 
						mean + "\t" + mini + "\t" + maxi);

	}	

	/**
	*/
	public void runSolution() {
		// Input image
		imp = WindowManager.getCurrentImage();
		if (imp == null) {
			IJ.error("Input image required");
			return;
		}
		
	 	if (checkImage() == false)
	 		return;
		
		// Parameters
		int nx = imp.getWidth();
		int ny = imp.getHeight();
		String operation = lstOperation.getSelectedItem();
	    int length = getIntegerValue( txtLength, 1, 3, 1000);
		if ( ((length-1)/2)*2 != (length-1)) {
			IJ.showMessage("The length should be odd.");
			return;
		}
		if ( length < 1) {
			IJ.showMessage("The length should greater than 1.");
			return;
		}
		if ( length >= nx || length >= ny) {
			IJ.showMessage("The length should smaller than the size.");
			return;
		}
		
		ImageStack inputstack=imp.getStack();
		int slices = inputstack.getSize();
		ImageStack stack = new ImageStack(nx, ny);
		ImageAccess out = null;
		String title = "[Teacher] ";
		String impl = "";
		String filter = "";
		
		// Processing
		double startTime = System.currentTimeMillis();
		for (int plane=1; plane <= slices; plane++) {
			ImageAccess image  = new ImageAccess (inputstack.getProcessor(plane)); 
		
			if (operation.equals("Vertical Edge Non-Separable")) {
				out = FilteringSolution.detectEdgeVertical_NonSeparable(image);
				impl = "non-sep";
				filter = "Vertical Edge";
				title += "Verti, Non-Sep";
			}  
			else if (operation.equals("Vertical Edge Separable")) {
				out = FilteringSolution.detectEdgeVertical_Separable(image);
				impl = "sep";
				filter = "Vertical Edge";
				title += "Verti, Sep";
			}  
			else if (operation.equals("Horizontal Edge Non-Separable")) {
				out = FilteringSolution.detectEdgeHorizontal_NonSeparable(image);
				impl = "non-sep";
				filter = "Horizontal Edge";
				title += "Horiz, Non-Sep";
			}  
			else if (operation.equals("Horizontal Edge Separable")) {
				out = FilteringSolution.detectEdgeHorizontal_Separable(image);
				impl = "sep";
				filter = "Horizontal Edge";
				title += "Horiz, Sep";
			}  
			else if (operation.equals("Moving-Average 5*5 Non-Separable")) {
				out = FilteringSolution.doMovingAverage5By5_NonSeparable(image);
				impl = "non-sep";
				title += "MA5x5, Non-Sep";
				filter = "Moving-Average 5*5";
				}  
			else if (operation.equals("Moving-Average 5*5 Separable")) {
				out = FilteringSolution.doMovingAverage5By5_Separable(image);
				impl = "sep";
				title += "MA5x5, Sep";
				filter = "Moving-Average 5*5";
			}  
			else if (operation.equals("Moving-Average 5*5 Recursive")) {
				out = FilteringSolution.doMovingAverage5By5_Recursive(image);
				title += "MA5x5, Recursive";
				impl = "recursive";
				filter = "Moving-Average 5*5";
			}  
			else if (operation.equals("Moving-Average L*L Recursive")) {
				out = FilteringSolution.doMovingAverageLByL_Recursive(image, length);
				title += "MALxL, Recursive";
				impl = "recursive";
				filter = "Moving-Average " + length + "*" + length;
			}  
			else if (operation.equals("Sobel Edge Detector")) {
				out = FilteringSolution.doSobel(image);
				title += "Sobel";
				impl = "";
				filter = "Sobel";
			}
			stack.addSlice("", out.createFloatProcessor());
		}

		ImagePlus impResult = new ImagePlus(title, stack);
		impResult.show();
		
		if (journal != null) {
			if (!journal.isVisible())
				journal = null;	
		}
		if (journal == null) {
			journal = new TextWindow("Journal", "Input Image\tFilter\tImpl.\tTime [ms]\tOutput Image\tMean(out)\tMin(out)\tMax(out)", "", 700, 250);
			Point loc = this.getLocation();
			Dimension dim = this.getSize();
			Point locj = journal.getLocation();
			journal.setLocation(locj.x, loc.y+dim.height+10);
			journal.show();
		}
		
		String time = IJ.d2s(System.currentTimeMillis() - startTime);
		String maxi = IJ.d2s(out.getMaximum());
		String mini = IJ.d2s(out.getMinimum());
		String mean = IJ.d2s(out.getMean());
		journal.append("" + imp.getTitle() + "\t" + filter + "\t" + impl + "\t" + time + "\t" + impResult.getTitle() + "\t" + 
						mean + "\t" + mini + "\t" + maxi);

	}	
	 
	/**
	* Check the size image.
	*/
	private boolean checkImage() {
		
		if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32) {
			IJ.showMessage("The image should be grayscale image 8-bits, 16-bits or 32-bits.");
			return false;
		}
		
		int nx = imp.getWidth();
		int ny = imp.getHeight();
			
		if ((nx < 3) || (ny < 3)) {
			IJ.showMessage("The image should be greater than 3.");
			return false;
		}
		return true;
	}

	/**
	* Add a component in a panel in the northeast of the cell.
	*/
	private void addComponent(Panel pn, int row, int col, int width, int height, int space, Component comp) {
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
	* Get a double value from a TextField between minimal and maximal values.
	*/
	private int getIntegerValue(TextField text, int mini, int defaut, int maxi) {
		int d;
		try {
			d = (new Integer(text.getText())).intValue();
			if (d < mini)  text.setText( "" + mini);
			if (d > maxi)  text.setText( "" + maxi);
		}
		
		catch (Exception e) {
			if (e instanceof NumberFormatException) 
				text.setText( "" + defaut);
		}
		d = (new Integer(text.getText())).intValue();
		return d;
	}
	
	/**
	* Implements the actionPerformed for the ActionListener.
	*/
	public synchronized  void actionPerformed(ActionEvent e) {
		if (e.getSource() == bnClose) {
			journal.hide();
			journal = null;
			dispose();
		}
		
		else if (e.getSource() == bnRunSession) {
			runSession();
		}
		
		else if (e.getSource() == bnRunSolution) {
			runSolution();
		}

		notify();
	}
	
	/**
	* Implements the itemStateChanged for the ItemListener.
	*/
	public synchronized void itemStateChanged(ItemEvent e) {
				
		if (e.getSource() == lstOperation){			
			String operation = lstOperation.getSelectedItem();
				if (operation == "Moving-Average L*L Recursive") {
					txtLength.show();
					lblLength.show();
				}
				else {
					txtLength.hide();
					lblLength.hide();
				}
		}
	}
	
	/**
	* Implements methods for the WindowListener.
	*/
	public void windowClosing(WindowEvent e) 		{ 
		dispose(); 
		journal.hide();
		journal = null;
	}
	public void windowActivated(WindowEvent e) 		{}
	public void windowClosed(WindowEvent e) 		{}
	public void windowDeactivated(WindowEvent e) 	{}
	public void windowDeiconified(WindowEvent e) 	{}
	public void windowIconified(WindowEvent e) 		{}
	public void windowOpened(WindowEvent e) 		{}

	
}

