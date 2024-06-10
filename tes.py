import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io

def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_dominant_colors(image, num_colors=5):
    image = image.resize((100, 100))  # Resize for faster processing
    img_data = np.array(image)
    img_data = img_data.reshape((img_data.shape[0] * img_data.shape[1], img_data.shape[2]))  # Reshape to list of RGB colors
    
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_data)
    
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def plot_colors(colors):
    fig, ax = plt.subplots(1, 1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.imshow([colors], aspect='auto')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

st.title("Dominant Color Picker")

st.write("Upload an image to get the dominant colors")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = load_image(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    colors = get_dominant_colors(image)
    st.write("Dominant Colors:")
    
    for i, color in enumerate(colors):
        hex_color = rgb_to_hex(color)
        st.write(f"Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]})")
        st.color_picker(f"Color Picker {i+1}", hex_color)
    
    buf = plot_colors(colors)
    st.image(buf, caption='Color Palette', use_column_width=True)
