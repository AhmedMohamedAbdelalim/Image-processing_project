import cv2
import numpy as np
import copy

def sobelOperator(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image file: {image_path}")
        return

    img = image.copy()
    Gx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    Gy = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    height, width = img.shape[:2]
    sobel_img = np.zeros_like(img, dtype=np.float32)

    for i in range(0, height-2):
        for j in range(0, width-2):
            gx = np.sum(np.multiply(Gx, img[i:i+3, j:j+3]))
            gy = np.sum(np.multiply(Gy, img[i:i+3, j:j+3]))
            sobel_img[i, j] = np.sqrt(gx ** 2 + gy ** 2)

    cv2.imwrite('sobelOperator.jpg', sobel_img)

image_path = 'E:/semester6/Image Processing (CS389CS486)/project/Ronaldo.jpg'
sobelOperator(image_path)


import numpy as np
import cv2

def fourier_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image file: {image_path}")
        return

    img = image.copy()
    height, width = img.shape[:2]
    copy_img = np.zeros_like(img, dtype=np.complex128)

    for i in range(height):
        for k in range(width):
            for x in range(height):
                for y in range(width):
                    copy_img[i, k] += img[x, y] * np.exp(-2j * np.pi * (i * x / height + k * y / width))
    
    copy_img = np.abs(copy_img)
    copy_img *= 255.0 / np.max(copy_img)  # Normalize the result
    copy_img = copy_img.astype(np.uint8)


image_path = 'E:/semester6/Image Processing (CS389CS486)/project/Ronaldo.jpg'
fourier_transform(image_path)


def gaussianNoise(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image file: {image_path}")
        return

    gaussian = np.random.normal(-1, 1, image.size)
    gaussian = gaussian.reshape(image.shape[0], image.shape[1]).astype('uint8')
    noisy_image = np.clip(image.astype(int) + gaussian.astype(int), 0, 255)

image_path = 'E:/semester6/Image Processing (CS389CS486)/project/Ronaldo.jpg'
gaussianNoise(image_path)


import numpy as np
import cv2

def roberts_cross(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image file: {image_path}")
        return

    img = image.copy()
    neighborhood_x = np.array([[-1, 0], [0, 1]])
    neighborhood_y = np.array([[0, -1], [1, 0]])

    height, width = img.shape[:2]

    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(1, height-1):
        for x in range(1, width-1):
            gx = np.sum(np.multiply(img[y-1:y+1, x-1:x+1], neighborhood_x))
            gy = np.sum(np.multiply(img[y-1:y+1, x-1:x+1], neighborhood_y))
            copy_img = int(np.sqrt(gx**2 + gy**2))
            output[y, x] = copy_img


image_path = 'E:/semester6/Image Processing (CS389CS486)/project/Ronaldo.jpg'
roberts_cross(image_path)
