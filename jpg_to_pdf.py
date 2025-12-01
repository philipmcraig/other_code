# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 20:14:53 2025

@author: phili
"""

import numpy as np
import os
from PIL import Image
import glob

folder = 'path/to/folder/'

files = glob.glob(folder+'*.jpg')

files_sort = list(np.concatenate([files[0:1],files[9:],files[1:9]]))

images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in files_sort]

output_pdf = "output.pdf"

images[0].save(folder+output_pdf, save_all=True, append_images=images[1:])