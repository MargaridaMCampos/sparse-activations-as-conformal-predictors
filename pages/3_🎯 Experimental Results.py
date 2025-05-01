import streamlit as st

st.set_page_config(
    #page_title='Sparse Activations as Conformal Predictors',
    #page_icon=':material/join_inner:',
    layout = 'wide',
    initial_sidebar_state= 'expanded',
    
)
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

#Styles 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('style.css')
# Plotly
config = {'displayModeBar': False}


with open("load-mathjax.js", "r") as f:
    js = f.read()
    st.components.v1.html(f"<script>{js}</script>", height=0)
    
st.markdown('# Experimental Results')
st.markdown('''
            We compared the proposed strategies with standard conformal prediction methods
            over different dimensions at several confidence levels.  
            The baseline strategies considered were: 
            * $\mathsf{InvProb}$: using $s(x,y)=1-\\text{softmax}_y(x)$,
            * $\mathsf{RAPS}$: *regularized adaptive prediction sets* strategy, introduced by [Angelopoulos et al., 2021](https://arxiv.org/pdf/2009.14193). ''')


'''### Efficiency  
An efficient conformal predictor should output small sets on average and 
singletons (prediction sets with a single label) with a high probability.  
#### Average Set Size'''
st.image('images/all_set_sizes_final.png')
'''#### Singleton Ratio'''
st.image('images/singleton_final.png')

'''### Adaptiveness  
Ideally, for any given partition of the data, we would have a coverage close to
the $1 − \\alpha$ bound. Analyzing the size-stratified coverage levels is a 
way to assess a predictor's adaptiveness.
'''

col1,col2 = st.columns([0.5,0.5])
with col1:
    st.image('images/imagenet.png', 
             caption='Size-stratified coverage for ImageNet data: $\\alpha=0.01$ (top) and $\\alpha=0.1$ (bottom).')
with col2:
    st.image('images/cifar100.png', caption='Size-stratified coverage for CIFAR100 data: $\\alpha=0.01$ (top) and $\\alpha=0.1$ (bottom).')

'''#### Size-stratified coverage violation '''
st.image('images/sscv.png', 
         caption='Size-stratified coverage violation (SSCV) for all methods and datasets, for $\\alpha \in \{0.01,0.05,0.1\}$')

explain = st.expander('More details',expanded=False)
with explain:
    ''' 
    Introduced by [Angelopoulos et al. (2021)](https://arxiv.org/pdf/2009.14193), SSCV (size-stratified coverage violation ) measures the maximum deviation from the desired coverage $1-\\alpha$.  
    Partitioning the possible size cardinalities into $G$ bins, $B_1, ..., B_G$, 
    let $\mathcal{I}_g$ be the set of observations falling in bin $g$, with $g = 1,...,G$, 
    the SSCV of a predictor $C_\\alpha$, for that bin partition is given by:'''
    st.latex('''\\begin{align}
        \\text{SSCV}(C, \{B_1,...,B_G\}) = \sup_g \left| \\frac{\left\{i : Y_i \in \mathcal{C}_\\alpha(X_i), i \in \mathcal{I}_g \\right\}}{|\mathcal{J}_j|} - (1 - \\alpha) \\right| \\nonumber
    \end{align}
    ''')
st.markdown('')
st.markdown('')