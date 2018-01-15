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
