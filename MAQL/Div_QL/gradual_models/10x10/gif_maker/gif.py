import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

img_1 = []
img_2 = []


z_no = 8
for i in range(1, z_no):

    img_1.append(Image.open("fig_1/DivQL_ratio_" + str(i) + ".png"))
    img_2.append(Image.open("fig_2/z_" + str(i+1) + ".png"))


for i in range(0, z_no-1):
    f, axarr = plt.subplots(2)
    axarr[0].set_title("z = " + str(i+1) )
    axarr[0].imshow(img_1[i])
    axarr[1].imshow(img_2[i])

    
    plt.savefig("fig_3/g_" + str(i))

img_3 = []
for i in range(0, z_no-1):
    img_3.append(imageio.imread("fig_3/g_" + str(i)+ ".png"))


imageio.mimsave('graph.gif', # output gif
                img_3,          # array of input frames
                fps = 1.5)

"""
with imageio.get_writer('graph.gif', mode='I') as writer:
    for image in img_3:
        image.show()
        writer.append_data(image)
"""