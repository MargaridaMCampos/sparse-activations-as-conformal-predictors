import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import altair as alt
import torch
from torch.nn.functional import softmax
from entmax.activations import sparsemax
from entmax.root_finding import entmax_bisect
import base64
#from streamlit_extras.app_logo import add_logo 
# Set page config
st.set_page_config(
    page_title='Sparse Activations as Conformal Predictors',
    page_icon=':material/join_inner:',
    layout = 'wide',
    initial_sidebar_state= 'expanded',
    
)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('style.css')
#col1, col2 = st.columns([0.7,0.3])
#with col1:
st.markdown('# Sparse Activations as Conformal Predictors')
col1, col2 = st.columns([0.7,0.3],gap='large')

with col1:
    st.markdown('''Our paper uncovers a novel connection between conformal prediction and sparse *softmax-like* transformations.''')
    st.markdown('''This website is meant to help the interested reader in understanding the core
                concepts used.  
                Check out the different pages:''')
    st.markdown(''' - [**Theory**](https://sparse-activations-conformal-predictors.streamlit.app/Theory) - interactive introductions
                to: conformal prediction; sparse activations and the **novel link** between them. 
                **Play with user inputs and see the connection in action.**''')
    st.markdown(''' - [**New Scores**](https://sparse-activations-conformal-predictors.streamlit.app/New_Scores) - short explanation
                of the new non-conformity scores introduced in our work.''')
    st.markdown(''' - [**Experimental Results**](https://sparse-activations-conformal-predictors.streamlit.app/Experimental_Results) - 
                summary of our experimental results.''')
    st.markdown(''' - [**About**](https://sparse-activations-conformal-predictors.streamlit.app/About) - author and project information.''')
with col2:
    st.markdown(
    """<a href="https://arxiv.org/pdf/2502.14773">
    <img src="data:image/png;base64,{}" width="400">
    </a>""".format(
        base64.b64encode(open("images/paper_thumbnail.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True)
    #st.image('images/paper_thumbnail.png',width=300,use_container_width=False)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
