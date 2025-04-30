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
st.write('# Sparse Activations as Conformal Predictors')
#with col2:
#    st.image('./images/logo.png',width=200)
# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
