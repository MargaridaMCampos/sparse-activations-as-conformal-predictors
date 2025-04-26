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
from streamlit_extras.app_logo import add_logo 
# Set page config
st.set_page_config(
    page_title='Sparse Activations as Conformal Predictors',
    page_icon=':material/join_inner:',
    layout = 'wide',
    initial_sidebar_state= 'expanded',
    
)

add_logo('./images/logo.png')
# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
st.write('# Sparse Activations as Conformal Predictors')
