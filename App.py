from ModelProcessor import ModelProcessor
import streamlit as st
import pandas as pd
from ModelProcessor import ModelProcessor

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


def app_globals():
    st.set_page_config('Dogs-cats classifier','wide')
    st.title('Model for classifying dogs and cats pictures')
    st.markdown('''### For more details about architecture, check [the source repo on GitHub](https://github.com/majoloso97/dog-cat-classsifier)
                ''')

app_globals()

url_refs = 'https://github.com/majoloso97/dog-cat-classsifier/blob/main/URL%20references.csv?raw=true'

model_url, example_names, example_urls = get_resources_data(url_refs)

model = initiate_model(model_url)

selected_image = st.selectbox('Choose an image to test the model:', example_names)

selected_url = example_urls[example_names.index(selected_image)]

st.write(model.show_predict_image(selected_url, from_url=True))