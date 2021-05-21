
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import efficientnet.tfkeras


with open("model\model.json") as json_file:
    json_savedModel= json_file.read()
#load the model architecture
model_j = tf.keras.models.model_from_json(json_savedModel,)

model_j.load_weights('model\model.h5')

#new_model = tf.keras.models.load_model('model')

### Excluding Imports ###
st.title("Upload Image Test")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    img = np.array(img)
    sq_size = min(img.shape[0], img.shape[1])
    img_r = tf.image.crop_to_bounding_box(img, np.int(round(img.shape[0] / 2 - sq_size / 2)),
                                          np.int(round(img.shape[1] / 2 - sq_size / 2)), sq_size, sq_size)
    img_r = np.int_(np.round(tf.image.resize(img_r, (224,224 )), 0))
    img_r = img_r.reshape(1,224,224,3)

    pred = model_j.predict(img_r/255.)


    #st.write("")
    if pred[0]>=0.5:
        st.title("Malignant")
    else:
        st.title("Begnin")
    st.image(img, caption='Uploaded Image.', width=100, use_column_width=True)


# left_column, right_column = st.beta_columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')
#
# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")