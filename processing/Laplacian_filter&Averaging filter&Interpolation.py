
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


def load_image():
    image = Image.open(
        "E:\semester6\Image Processing (CS389CS486)\project\Ronaldo.jpg")
    image = image.resize((300, 300))
    # Create a PhotoImage object from the image
    photo = ImageTk.PhotoImage(image)

    # Create a label widget to display the image
    label = tk.Label(window, image=photo)
    label.image = photo  # Store a reference to the image
    # Place the label in the window
    label.pack()


def apply_kernel(image, kernel):
    image_height, image_width = image.shape[:2]
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            filtered_value = np.sum(region * kernel)
            result[i, j] = filtered_value

    return result


def laplacian_filter(image):

    # Convert the PIL image to a NumPy array
    np_image = np.array(image)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    filtered_image =apply_kernel(np_image,kernel)
    # Convert the filtered image to uint8 format
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    # Convert the filtered image back to a PIL image
    filtered_image = Image.fromarray(filtered_image)
    return filtered_image


def average_filter(image):
    np_image = np.array(image)
    output = np.zeros_like(np_image)
    m, n = np_image.shape[:2]
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9

    for i in range(1, m-1):
        for j in range(1, n-1):
            AVG_filter = np_image[i-1, j-1]*mask[0, 0]+np_image[i-1, j]*mask[0, 1]+np_image[i-1, j + 1]*mask[0, 2]+np_image[i, j-1]*mask[1, 0]+np_image[i,
                                                                                                                                                        j]*mask[1, 1]+np_image[i, j + 1]*mask[1, 2]+np_image[i + 1, j-1]*mask[2, 0]+np_image[i + 1, j]*mask[2, 1]+np_image[i + 1, j + 1]*mask[2, 2]
            output[i, j-1] = AVG_filter

    return Image.fromarray(output.astype(np.uint8))


def Interpolation(image):
    np_image = np.array(image)
    m, n = np_image.shape[:2]
    for i in range(m):
        row = np_image[i, :]
        np_image = np.insert(np_image, 2*i+1, row, axis=0)

    np_image1 = np_image

    for j in range(n):
        column_name = np_image[:, j]
        np_image1 = np.insert(np_image1, 2*j+1, column_name, axis=1)
    pil_image = Image.fromarray(np_image1)
    desired_width = 222
    desired_height = 222
    resized_image = pil_image.resize((desired_width, desired_height))
    return pil_image


def handle_selection(event):
    selected = combo_box_1.get()
    image = Image.open(
 "E:\semester6\Image Processing (CS389CS486)\project\Ronaldo.jpg")
    if selected == "Laplacian Operator":
        filtered_image = laplacian_filter(image)
    elif selected == "Averaging filter":
        filtered_image = average_filter(image)
    elif selected == "Interpolation":
        filtered_image = Interpolation(image)

    # Create a PhotoImage object from the filtered image
    photo = ImageTk.PhotoImage(filtered_image)

    # Create a label widget to display the image
    label = tk.Label(window, image=photo)
    label.image = photo  # Store a reference to the image

    # Place the label in the window
    label.pack(side="left")


window = tk.Tk()

# Create the main window
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the window geometry to the screen resolution
window.geometry(f"{screen_width}x{screen_height}")

# Set the background color of the window to blue
window.configure(bg="skyblue")

# Load and display the image
load_image()

# Create the combobox
combo_box_1 = ttk.Combobox(window, width=30, font=('Arial', 10, 'bold'))
combo_box_1['values'] = (
    'Laplacian Operator',
    'Averaging filter',
    'Interpolation',
)
combo_box_1.pack()

# Bind the event handler to the combobox
combo_box_1.bind("<<ComboboxSelected>>", handle_selection)

# Start the main event loop
window.mainloop()
