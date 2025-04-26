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

# Set page config
st.set_page_config(
    page_title='Sparse Activations as Conformal Predictors',
    page_icon=':material/join_inner:',
    layout = 'wide',
    initial_sidebar_state= 'expanded'
)
st.sidebar.title('Hi')
# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
st.write('# Sparse Activations as Conformal Predictors')
