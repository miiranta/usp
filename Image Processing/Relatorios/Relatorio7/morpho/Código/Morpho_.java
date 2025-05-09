import java.awt.*;
import java.awt.event.*;
import java.util.*;
import ij.*;
import ij.gui.*;
import ij.process.*;


public class Morpho_ extends Dialog implements ActionListener, ItemListener {


	private ImagePlus imp;			// Reference to the input image
	private ImageStack stack;			// Reference to the output stack

	private String	operation="";			// Kind of operator
	private int iterations=3;				// Number of iterations
	private int connectivity=8;				// Type of structuring element (8 or 4 connected)

	private Choice lstOperation;
	private Button bnOK;
	private Button bnCancel;
	private Choice lstIterations;
	private Choice lstConnectivity;
	private int[] wList;
	private Vector index = new Vector(1,1);


	/**
	* Constructor.
	*/
	public Morpho_() {
		super(new Frame(), "Morphological Operators");
		
		if (IJ.versionLessThan("1.21a"))
			return;
		
		doDialog();
			
	}
	
	/**
	* Run.
	*/
	public void run() {
		imp = WindowManager.getCurrentImage();
		if (imp == null) {
			IJ.error("Input image required");
			return;
		}
		
	 	if (checkSizeImage() == false)
	 		return;
	 	
		// Processing
		int nx = imp.getWidth();
		int ny = imp.getHeight();
		ImageStack inputstack=imp.getStack();
		int size = inputstack.getSize();
		stack = new ImageStack(nx, ny);
		double startTime = System.currentTimeMillis();

		for (int plane=1; plane<=size; plane++) {

			ImageAccess image = new ImageAccess(inputstack.getProcessor(plane));
			ImageAccess output = null;
			
			for (int n=0; n<iterations; n++) {
				// Erosion
				if (operation == "Min")
					if (connectivity == 4)
						output = Morpho4Solut.doErosion(image);
					else
						output = Morpho8.doErosion(image);
				// Dilation
				if (operation == "Max")
					if (connectivity == 4)
						output = Morpho4Solut.doDilation(image);
					else
						output = Morpho8.doDilation(image);
				// Open
				if (operation == "Open")
					if (connectivity == 4)
						output = Morpho4Solut.doOpen(image);
					else
						output = Morpho8.doOpen(image);
				// Close
				if (operation == "Close")
					if (connectivity == 4)
						output = Morpho4Solut.doClose(image);
					else 
						output = Morpho8.doClose(image);
						
				// Gradient
				if (operation == "Gradient")
					if (connectivity == 4)
						output = Morpho4Solut.doGradient(image);
					else
						output = Morpho8.doGradient(image);
				// Top hat - bright feature detection
				if (operation == "Top hat bright")
					if (connectivity == 4)
						output = Morpho4Solut.doTopHatBright(image);
					else
						output = Morpho8.doTopHatBright(image);
				// Top hat - dark feature detection
				if (operation == "Top hat dark")
					if (connectivity == 4)
						output = Morpho4Solut.doTopHatDark(image);
					else
						output = Morpho8.doTopHatDark(image);
				// Median
				if (operation == "Median") 
					if (connectivity == 4)
						output = Morpho4Solut.doMedian(image);
					else
						output = Morpho8.doMedian(image);
				image = output.duplicate();
			}
			// Store into the the stack result
			stack.addSlice("", output.createFloatProcessor());
		}

		// Display
		String title = (new Integer(connectivity)).toString() + "-connect " + operation;
		ImagePlus impResult = new ImagePlus(title, stack);
		impResult.show();
		impResult.updateAndDraw();
	}	


	/**
	* Check the size image.
	*/
	private boolean checkSizeImage() {
		int nx = imp.getProcessor().getWidth();
		int ny = imp.getProcessor().getHeight();
			
		if ((nx < 3) || (ny < 3)) {
			IJ.showMessage("The image should be greater than 3.");
			return false;
		}
		return true;
	}

	/**
	* Check the parameters.
	*/
	private boolean checkParameters() {
		return true;
	}


	/**
	* Build the dialog box.
	*/
	private void doDialog()	{
		GridBagLayout grid = new GridBagLayout();
		GridBagConstraints c = new GridBagConstraints();
		setLayout(grid);		
		lstOperation = new Choice();
		lstOperation.addItem("Min");
		lstOperation.addItem("Max");
		lstOperation.addItem("Open");
		lstOperation.addItem("Close");
		lstOperation.addItem("Gradient");
		lstOperation.addItem("Top hat bright");
		lstOperation.addItem("Top hat dark");
		lstOperation.addItem("Median");
		lstOperation.select(1);
		
		lstIterations = new Choice();
		lstIterations.addItem("1");
		lstIterations.addItem("2");
		lstIterations.addItem("3");
		lstIterations.addItem("4");
		lstIterations.addItem("5");
		lstIterations.addItem("6");
		lstIterations.addItem("7");
		lstIterations.addItem("8");
		lstIterations.addItem("9");
		lstIterations.addItem("10");
		lstIterations.select(0);
			
	 	lstConnectivity = new Choice();
	 	lstConnectivity.add("4-connected");
		lstConnectivity.add("8-connected");
		lstConnectivity.select(1);
		
		Panel pnInput = new Panel();
		pnInput.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 0));


		buildCell(grid, c, 1, 0, 1, 1, pnInput);
		buildCell(grid, c, 0, 1, 1, 1, new Label("Operation"));
		buildCell(grid, c, 1, 1, 1, 1, lstOperation);
		buildCell(grid, c, 0, 2, 1, 1, new Label("Iterations"));
		buildCell(grid, c, 1, 2, 1, 1, lstIterations);
		buildCell(grid, c, 0, 3, 1, 1, new Label("Connectivity"));
		buildCell(grid, c, 1, 3, 1, 1, lstConnectivity);

		
		// Panel buttons 
		Panel buttons = new Panel();
		buttons.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 0));
		bnCancel = new Button("Close");
		buttons.add(bnCancel);
		bnOK = new Button("  Run  ");
		buttons.add(bnOK);
	    c.gridx = 1;
	    c.gridy = 5;
	    c.gridwidth = 1;
	    c.gridheight = 1;
	    c.anchor = GridBagConstraints.EAST;
	    c.insets = new Insets(20, 5, 5, 5);
	    grid.setConstraints(buttons, c);
	    add(buttons);
		

		// Event Handler
		
		bnOK.addActionListener(this);
		bnCancel.addActionListener(this);
		lstOperation.addItemListener(this);

		// Building the main panel
		
		pack();
		GUI.center(this);
		setVisible(true);
		Point pos = getLocation();
		setLocation(pos.x+300, pos.y+100);
		IJ.wait(250); 	// work around for Sun/WinNT bug

	}

	/**
	* Build one cell of the dialog box.
	*/
	private void buildCell(GridBagLayout gbl, GridBagConstraints gbc, int x, int y, int w, int h, Component Comp) {
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    gbc.anchor = GridBagConstraints.NORTHWEST;
	    gbc.insets = new Insets(5, 5, 5, 5);
	    gbl.setConstraints(Comp, gbc);
	    add(Comp);
	}


	/**
	* Implements the actionPerformed for the ActionListener.
	*/
	public synchronized  void actionPerformed(ActionEvent e) {
		if (e.getSource() == bnCancel) {
			dispose();
		}
		
		if (e.getSource() == bnOK) {
			operation = (String)lstOperation.getSelectedItem();
			iterations = lstIterations.getSelectedIndex()+1;
			if ((String)lstConnectivity.getSelectedItem() == "4-connected")
				connectivity = 4;
			else
				connectivity = 8;

			run();
			
		
			
		}

		notify();
	}
	/**
	* Implements the itemStateChanged for the ItemListener.
	*/
	public synchronized void itemStateChanged(ItemEvent e){
					
		if (e.getSource() == lstOperation){			
			operation = (String)lstOperation.getSelectedItem();
			
		}
	}
}
