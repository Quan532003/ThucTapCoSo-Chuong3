import numpy as np
import matplotlib.pyplot as plt

def loadImageFile(fileimage):
    f = open(fileimage, "rb")
    f.read(16)
    pixels = 28*28
    images_arr = []
    while True:
        try:
            img = []
            for j in range(pixels):
                pix = ord(f.read(1))
                img.append(pix / 255)
            images_arr.append(img)
        except:
            break
    f.close()
    image_sets = np.array(images_arr)
    return image_sets

test_images = loadImageFile("t10k-images.idx3-ubyte/t10k-images.idx3-ubyte")

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()
    return

gen_image(test_images[0])


def loadLabelFile(filelabel):
    f = open(filelabel, "rb")
    f.read(8)
    labels_arr = []
    while True:
        row = [0 for x in range(10)]
        try:
            label = ord(f.read(1))
            row[label] = 1
            labels_arr.append(row)
        except:
            break
    f.close()
    label_sets = np.array(labels_arr)
    return label_sets

test_labels = loadLabelFile("t10k-labels.idx1-ubyte")
print(test_labels[0])


