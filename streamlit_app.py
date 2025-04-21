import streamlit as st
import pandas as pd
import math
import numpy as np
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Sparse Activations as Conformal Predictors',
    page_icon=':material/join_inner:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :material/join_inner: Sparse Activations as Conformal Predictors


'''

# Add some spacing
''
''
''' ## Theoretical Setup

'''
tab1, tab2, tab3 = st.tabs(["📊 Conformal Prediction", "📈 Sparse Activations","🔗 Equivalence"])
data = np.random.randn(10, 1)

tab1.subheader("A tab with a chart")
tab1.line_chart(data)

tab2.subheader("Sparse Alternatives to Softmax")
tab2.write('''
           In a classification task with $K$ classes, 
           typical predictive models output a vector of label scores $\mathbf{z}\in \mathbb{R}^K$,
           which is converted to a probability vector through some transformation - typically *softmax*.
           
           [Martins and Astudillo (2016)](https://arxiv.org/pdf/1602.02068) introduced *sparsemax* -
           an alternative transformation capable of producing sparse outputs.
           [Peters et al. (2019)](https://arxiv.org/pdf/1905.05702) showed that both transformations
           are particular cases of the $\gamma\mathrm{-entmax}$ family
           ''')

tab3.subheader("A tab with the data")
tab3.write(data)