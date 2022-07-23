import streamlit as st
import pandas as pd
from PIL import Image
from ModelProcessor import ModelProcessor
from io import BytesIO

@st.experimental_singleton
def initiate_model(source_url):
    return ModelProcessor(source_url)


@st.experimental_memo
def get_resources_data(csv_url):
    data = pd.read_csv(csv_url)
    model_url = data[data['AssetType'] == 'model']['URL'][0]

    example_names = list(data[data['AssetType'] != 'model']['Name'])
    example_urls = list(data[data['AssetType'] != 'model']['URL'])
    return model_url, example_names, example_urls


@st.experimental_memo
def memoize_examples(example_names, example_urls, _model):
    examples = {}
    for i in range(len(example_names)):
        examples[example_names[i]] = _model.show_predict_image(example_urls[i], source_type = 'url')
    return examples


def app_globals():
    st.set_page_config('Dogs-cats classifier',layout='wide')
    st.title('Model for classifying dogs and cats pictures')
    st.markdown('''### For more details about architecture, check [the source repo on GitHub](https://github.com/majoloso97/dog-cat-classsifier)
                ''')

app_globals()

url_refs = 'https://github.com/majoloso97/dog-cat-classsifier/blob/main/URL%20references.csv?raw=true'

model_url, example_names, example_urls = get_resources_data(url_refs)

model = initiate_model(model_url)
examples = memoize_examples(example_names, example_urls, model)

with st.expander('Examples'):
    A, B = st.columns([3,1])

    selected_image = A.selectbox('Choose an image to test the model:', example_names)
    prediction, plot = examples[selected_image]

    A.subheader('The selected image is' + prediction)
    B.subheader("Loaded image:")
    B.pyplot(plot)

with st.expander('Predict from camera'):
    C, D = st.columns(2)
    pic = C.camera_input('Take a photo!')
    if pic is not None:
        img = Image.open(pic)
        prediction, plot = model.show_predict_image(img, source_type='camera')
        D.subheader('The captured photo is:' + prediction)
        D.subheader("Input image to model:")
        D.pyplot(plot)

with st.expander('Upload file from computer'):
    E, F = st.columns(2)
    pic = E.file_uploader('Upload an image (only jpeg allowed)', ['jpeg', 'jpg'])
    if pic is not None:
        img = Image.open(BytesIO(pic.getvalue()))
        prediction, plot = model.show_predict_image(img, source_type='camera')
        E.subheader('The captured photo is:' + prediction)
        F.subheader("Input image to model:")
        F.pyplot(plot)