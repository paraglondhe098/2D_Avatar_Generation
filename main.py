from utils import ImageGenerator, load_generator
from models import Generator
import numpy as np
import streamlit as st

if __name__ == '__main__':

    labels_dict = {'eye_angle': 0,
                   'eye_lashes': 0,
                   'eye_lid': 0,
                   'chin_length': 2,
                   'eyebrow_weight': 0,
                   'eyebrow_shape': 0,
                   'eyebrow_thickness': 0,
                   'face_shape': 2,
                   'facial_hair': 4,
                   'hair': 74,
                   'eye_color': 2,
                   'face_color': 5,
                   'hair_color': 3,
                   'glasses': 11,
                   'glasses_color': 6,
                   'eye_slant': 0,
                   'eyebrow_width': 0,
                   'eye_eyebrow_distance': 0}
    sizes = (3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3)

    for label, size in zip(labels_dict.keys(), sizes):
        labels_dict[label] = st.sidebar.slider(" ".join(label.capitalize().split("_")), 0, size-1)

    gen = Generator()
    gen.load_state_dict(load_generator())
    ig = ImageGenerator(gen)
    images = [ig.generate(labels_dict) for _ in range(16)]
    cols = st.columns(4)
    for i, image in enumerate(images):
        with cols[i % 4]:
            st.image(image, caption=f'Image {i + 1}', width=128)

    # st.image(image, caption='Your Avatar', width=256)
