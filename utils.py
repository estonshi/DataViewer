import numpy as np
import sys

def processImage(infiles, savepath):
	from PIL import Image
	if len(infiles)<2:
		return "Not enough figures."
	im = Image.open(infiles[0])
	images = []
	for f in infiles[1:]:
		images.append(Image.open(f))
	im.save(savepath, save_all=True, append_images=images, loop=1, duration=1)

def generateAVI(infiles, savepath):
	try:
		import cv2
	except:
		from traitsui.message import message
		message("You need to install opencv2 first! Use 'conda install'.")
		return

	frame = cv2.imread(infiles[0])
	height, width, layers = frame.shape
	video = cv2.VideoWriter(savepath, -1, 1, (width,height))
	for f in infiles[1:]:
		video.write(cv2.imread(f))

	cv2.destroyAllWindows()
	video.release()