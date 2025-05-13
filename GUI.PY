import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def load_image():
    # Open the image file
    image = Image.open("E:\semester6\Image Processing (CS389CS486)\project\download.jpg")
    
    # Resize the image if needed
    image = image.resize((300, 300))
    
    # Create a PhotoImage object from the image
    photo = ImageTk.PhotoImage(image)
    
    # Create a label widget to display the image
    label = tk.Label(window, image=photo)
    label.image = photo  # Store a reference to the image
    
    # Place the label in the window
    label.pack()

# Create the main window
window = tk.Tk()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()


# Set the window geometry to the screen resolution
window.geometry(f"{screen_width}x{screen_height}")

# Set the background color of the window to blue
window.configure(bg="blue")

# Load and display the image
load_image()






def handle_selection(event):
    selected =combo_box_1.get()
    if(selected=="Alamir"):
         import ahmed
    if(selected=="Ali"):
         import Ali
    if(selected=="Sakr"):
         import Sakr
    if(selected=="George"):
         import george
        
         # Hide the window
         window.withdraw()
   
    # if(selected=="Sharpening Spatial filters"):
    # if(selected=="'Noise filters"):
    # if(selected=="Frequency Domain filters"):
         

# Create a combo box
# Create the first-level combo box
combo_box_1 = ttk.Combobox(window)
combo_box_1['values'] = ('Alamir', 'Ali', 'Sakr', 'George')
combo_box_1.pack()
combo_box_1.pack(pady=10)
combo_box_1.configure(font=('Arial', 12, 'bold'))
#combo_box_1.pack(font=('Arial', 12, 'bold'))

# Bind the event handler to the combobox

combo_box_1.bind("<<ComboboxSelected>>", handle_selection)
# # Create a dictionary for the second-level options
# color_options = {
#     'Smoothing Spatial filters': ('Median filter', 'Adaptive filter', 'Averaging filter','Gaussian filter'),
#     'Sharpening Spatial filters': ('Laplacian Operator', 'Unsharp Masking and Highboost Filtering', 'Roberts Cross-Gradient Operators','Sobel Operators'),
#    
#     'Frequency Domain filters': ('Histogram Equalization', 'Histogram Specification', 'Fourier transform','Interpolation ')

# }





# # Create the second-level combo box
# combo_box_2 = ttk.Combobox(window)
# combo_box_2.pack(pady=10)
# combo_box_2.configure(font=('Arial', 12, 'bold'))
# combo_box_2.pack()

# def update_combo_box_2(event):
#     selected_color = combo_box_1.get()
#     combo_box_2['values'] = color_options[selected_color]
#     print(  combo_box_2['values'])
# #combo_box_1.bind("<<ComboboxSelected>>", update_combo_box_2)
# style = ttk.Style()
# style.configure('TCombobox', padding=5, relief='solid', foreground='black', background='white', fieldbackground='yellow')
# style.configure('TCombobox',
#                 padding=5,
#                 relief='solid',
#                 foreground='black',
#                 background='white',
#                 fieldbackground='yellow',
#                 font=('Arial', 12, 'bold'))
# style.map('TCombobox', fieldbackground=[('readonly', 'white')])

# Start the main event loop
window.mainloop()
