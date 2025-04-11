import ij.*;
import ij.gui.*;
import ij.process.*;
import java.awt.*;
import java.awt.event.*;

/*
 * Tomography Session class.
 *
 * @version 1.0
 *
 */

public class Radon_ extends Dialog
	implements ActionListener, ItemListener
{
private int 	Nangles=128;
private int 	Deg=0;
private String	operation="Radon Transform";
private	String  filter="Ram-Lak";

private ImagePlus impSource;
private ImageWindow wSource;
private ImageWindow wResult;
private ImageWindow wGraph;

/**
* Constructor.
*/
public Radon_() 
{
	super(new Frame(), "Tomography");
	
	if (IJ.versionLessThan("1.21a"))
		return;
		
	doDialog();
}

/**
* Process the image.
*/
public void runSession()
{
	ImagePlus imp = WindowManager.getCurrentImage();
	 	
	// Get the parameters
	String filter = choiceFilter.getSelectedItem(); 
	int Nangles = (new Integer(choiceAngle.getSelectedItem()).intValue());
	
	// Processing
	ImageAccess original = new ImageAccess(imp.getProcessor());
	ImageAccess sino = null;
	ImageAccess sinoF = null;
	ImageAccess output = null;
	ImageAccess image = original.duplicate();
	
	// Direct Transform
	if (chkRadonTransform.getState() == true) {
		if (checkImage(imp) == false)
			return;
		sino = Radon.transformRadon(image, Nangles);
		sino.show("Radon Transform");
	} 
	else {
		sino = image.duplicate();
	} 
	
	// Filtering
	if (chkFilter.getState() == false)
	    sinoF = sino.duplicate();
	else if (filter.equals("Ram-Lak"))
		sinoF = Radon.applyRamLakFilter(sino);
	else if (filter.equals("Cosine")) 
		sinoF = Radon.applyCosineFilter(sino);
	else if (filter.equals("Laplacian"))
		sinoF = Radon.applyLaplacianFilter(sino);
	if (chkFilter.getState() == true)
		sinoF.show(filter);	
		
	// BackProjection
	if (chkBackprojection.getState() == true) {
		output = Radon.inverseRadon(sinoF);
		output.show("Reconstructed");
	}
}

/**
* Process the image.
*/
public void runSolution()
{
	// Get and check the image	
	ImagePlus imp = WindowManager.getCurrentImage();
 	
	// Get the parameters
	String filter = choiceFilter.getSelectedItem(); 
	int Nangles = (new Integer(choiceAngle.getSelectedItem()).intValue());
	
	// Processing
	ImageAccess original = new ImageAccess(imp.getProcessor());
	ImageAccess sino = null;
	ImageAccess sinoF = null;
	ImageAccess output = null;
	ImageAccess image = original.duplicate();
	
	// Direct Transform
	if (chkRadonTransform.getState() == true) {
		if (checkImage(imp) == false)
			return;
		sino = RadonSolution.transformRadon(image, Nangles);
		sino.show("Radon Transform");
	} 
	else {
		sino = image.duplicate();
	} 
	
	// Filtering
	if (chkFilter.getState() == false)
	    sinoF = sino.duplicate();
	else if (filter.equals("Ram-Lak"))
		sinoF = RadonSolution.applyRamLakFilter(sino);
	else if (filter.equals("Cosine")) 
		sinoF = RadonSolution.applyCosineFilter(sino);
	else if (filter.equals("Laplacian"))
		sinoF = RadonSolution.applyLaplacianFilter(sino);
	if (chkFilter.getState() == true)
		sinoF.show(filter);	
		
	
	// BackProjection
	if (chkBackprojection.getState() == true) {
		output = RadonSolution.inverseRadon(sinoF);
		output.show("Reconstructed");
	}
}

/**
* Build the dialog box.
*/
private GridBagLayout 		layout;
private GridBagConstraints 	constraint;
private Checkbox 			chkRadonTransform;
private Checkbox 			chkFilter;
private Checkbox 			chkBackprojection;
private Choice 				choiceAngle;
private Choice 				choiceFilter;
private Button 				bnClose;
private Button 				bnRunSession;
private Button 				bnRunSolution;
private TextField			txtFactor;
private Label				lblAngle;

private void doDialog()
{
	// Layout
	layout = new GridBagLayout();
	constraint = new GridBagConstraints();
	
	// Check action
	chkRadonTransform = new Checkbox("Radon Transform", true);
	chkFilter = new Checkbox("Filtering sinogram", false);
	chkBackprojection = new Checkbox("Backprojection sinogram", false);
	
	// Angle
	Panel pnAngle = new Panel();
	lblAngle = new Label("Number of angles", Label.RIGHT);
	choiceAngle = new Choice();
	choiceAngle.add("512"); 
	choiceAngle.add("256"); 
	choiceAngle.add("128"); 
	choiceAngle.add("64"); 
	choiceAngle.add("32"); 
	choiceAngle.add("16"); 
	choiceAngle.add("8"); 
	choiceAngle.add("4"); 
	choiceAngle.select(2);
	pnAngle.setLayout(new FlowLayout(FlowLayout.LEFT, 3, 0));
	pnAngle.add(lblAngle);
	pnAngle.add(choiceAngle);

	// Choice Filtering
	choiceFilter = new Choice();
	choiceFilter.add("Ram-Lak");
	choiceFilter.add("Cosine");
	choiceFilter.add("Laplacian");
	choiceFilter.select(0);

	// Buttons
	bnClose 	   = new Button("Close");
	bnRunSession   = new Button("Run Session");
	bnRunSolution  = new Button("Run Solution");

	// Panel buttons 
	Panel pnButtons = new Panel();
	pnButtons.setLayout(new FlowLayout(FlowLayout.RIGHT, 8, 0));
	pnButtons.add(bnRunSolution);
	pnButtons.add(bnClose);
	pnButtons.add(bnRunSession);

	// Panel parameters	
	Panel pnParameters = new Panel();
	pnParameters.setLayout(layout);
 	addComponent(pnParameters, 0, 0, 1, 1, 5, chkRadonTransform);
 	addComponent(pnParameters, 0, 1, 1, 1, 5, pnAngle);
 	addComponent(pnParameters, 1, 0, 1, 1, 5, chkFilter);
 	addComponent(pnParameters, 1, 1, 1, 1, 5, choiceFilter);
	addComponent(pnParameters, 5, 0, 1, 1, 5, chkBackprojection);
	addComponent(pnParameters, 7, 0, 2, 1, 5, pnButtons);	
		
	// Add Listeners
	bnClose.addActionListener(this);
	bnRunSession.addActionListener(this);
	bnRunSolution.addActionListener(this);
	chkRadonTransform.addItemListener(this);
	chkFilter.addItemListener(this);
	chkBackprojection.addItemListener(this);
	choiceFilter.addItemListener(this);
	
	// Building the main panel
	add(pnParameters);
	pack();
	setResizable(false);
	GUI.center(this);
	setVisible(true);
	IJ.wait(250); 	// work around for Sun/WinNT bug
	enableComponents();
}

/**
* Add a component in a panel in the northeast of the cell.
*/
final private void addComponent(
				final Panel pn,
				final int row, final int col, 
				final int width, final int height, 
				final int space,
				final Component comp)
{
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
public synchronized  void actionPerformed(ActionEvent e) 
{
	if (e.getSource() == bnClose) {
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
public synchronized void itemStateChanged(ItemEvent e)
{
	enableComponents();
}

/**
* Set the status of the compoments.
*/
private void enableComponents()
{
	if (!chkRadonTransform.getState() &&  !chkFilter.getState() && !chkBackprojection.getState()) {
		bnRunSession.disable();
		bnRunSession.disable();
	}	
	else {
		bnRunSession.enable();
		bnRunSession.enable();
	}	
	if (chkFilter.getState() == false)
		 choiceFilter.hide();
	else
		 choiceFilter.show();
}

private boolean checkImage(ImagePlus imp)
{
	int nx = imp.getWidth();
	int ny = imp.getHeight();
	if (nx != ny) {
		IJ.error("The input image should be square.");
		return false;
	}

	int i=2;
	while(i<nx) i *= 2;
	if (i != nx) {
		IJ.error("The size of the image should be a power of 2.");
		return false;
	}
	return true;
}

} // end of class

