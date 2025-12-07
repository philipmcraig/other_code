# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 20:14:53 2025

@author: phili
"""

import numpy as np
import os
from PIL import Image
import wget
import cairosvg
import glob

folder = 'C:/Users/phili/Documents/imfc_128/'
url = 'https://svg.issuu.com/251124110432-51a3ee0a63f124e8e3ef80ca9d3b1898/'

for i in range(1,18):
    wget.download(url+'page_'+str(i)+'.svg',out=folder)

files_svg = glob.glob(folder+'*.svg')
files_svg_sort = list(np.concatenate([files_svg[0:1],files_svg[9:],files_svg[1:9]]))

for i in range(len(files_svg_sort)):
    png_file = 'temp_'+str(i+1)+'.png'
    jpg_file = 'page_'+str(i+1)+'.jpg'
    cairosvg.svg2png(url=files_svg_sort[i], write_to=folder+png_file)
    img = Image.open(folder+png_file).convert("RGB")
    img.save(folder+jpg_file, "JPEG", quality=95)

files = glob.glob(folder+'page*.jpg')

files_sort = list(np.concatenate([files[0:1],files[9:],files[1:9]]))

images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in files_sort]

output_pdf = "output.pdf"

images[0].save(folder+output_pdf, save_all=True, append_images=images[1:])