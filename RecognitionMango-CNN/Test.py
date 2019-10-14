# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2
 
def select_image():
	# grab a reference to the image panels
	global panelA, panelB
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkinter.filedialog.askopenfilename()
	# ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		
 
		# OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
		# convert the images to PIL format...
		image = Image.fromarray(image)
		
 
		# ...and then to ImageTk format
		image = ImageTk.PhotoImage(image)
		
		# if the panels are None, initialize them
		if panelA is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
 
			
		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image)
			
			panelA.image = image
			

			# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None

 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()