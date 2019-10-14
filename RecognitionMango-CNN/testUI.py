# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
from imutils import paths
import numpy as np
import pickle
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2
import os
 
def select_image():

	# grab a reference to the image panels
	global panelA
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkinter.filedialog.askopenfilename()
	# ensure a file path was selected
	if len(path) > 0:
		image = cv2.imread(path)
		output = image.copy()
		 
		# pre-process the image for classification
		image = cv2.resize(image, (96, 96))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# load the trained convolutional neural network and the label
		# binarizer
		model = load_model('RecognitionMango_60_round.model')
		lb = pickle.loads(open('Mango.pickle', "rb").read())

		proba = model.predict(image)[0]
		idx = np.argmax(proba)
		label = lb.classes_[idx]

		# we'll mark our prediction as "correct" of the input image filename
		# contains the predicted label text (obviously this makes the
		# assumption that you have named your testing image files this way)
		#filename = [image][[image].rfind(os.path.sep) + 1:]
		#correct = "correct" if filename.rfind(label) != -1 else "incorrect"

		# build the label and draw the label on the image
		label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, "correct")
		output = imutils.resize(output, width=400)
		cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 255, 0), 2)

		#output = Image.fromarray(output)
		#output = ImageTk.Photoimage(output)
		#panelA = Label(image=output)
		#panelA.output = output
		#panelA.pack(padx=10, pady=10)
		#panelA.configure(image=output)
		#panelA.output = output

		# show the output image
		#print("[INFO] {}".format(label))
		#cv2.imshow("Output", output)
		#cv2.waitKey(0)
		# OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
 
		# convert the images to PIL format...
		output = Image.fromarray(output)
		
 
		# ...and then to ImageTk format
		output = ImageTk.PhotoImage(output)
		
		# if the panels are None, initialize them
		if panelA is None:
			# the first panel will store our original image
			panelA = Label(image=output)
			panelA.output = output
			panelA.pack( padx=1, pady=1)
			#panelA.pack(width = 50, height = 100)
			
		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=output)
			
			panelA.output = output
			
		

root = Tk()
root.title("Output")
##############################

panelA = None

#############################
#listbox = Listbox(root)
#listbox.pack(ipadx=128, ipady=168)

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="2", pady="2")
 
# kick off the GUI
root.mainloop()