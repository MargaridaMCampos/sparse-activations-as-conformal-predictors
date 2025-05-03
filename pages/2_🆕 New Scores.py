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
    
st.markdown('# Novel Scores ')

st.markdown('''Uncovering the relationship between conformal prediction and
            temperature scaling of $\gamma$-entmax motivated the introduction
            of new non-conformity scores for conformal classification:''')

#st.markdown('''## New''')

col1,col2,col3 = st.columns(3,border=True,gap='medium')
with col1:
    st.markdown(''' #### $\gamma\\text{-}\mathsf{entmax}$''')
    st.markdown('''Score equivalent to the temperature scaling of $\gamma$-entmax:''')
    scol1, scol2, scol3 = st.columns([0.1,0.8,0.1])
    with scol1:
        st.write("")
    with scol2:
        st.markdown('''$$s(x,y) = \|\\bm{z}_{1:k(y)}-z_{k(y)}\mathbf{1}\|_\delta,$$''')
    with scol3:
        st.write("")
    st.markdown("")
    st.markdown('''with $\delta=\\frac{1}{\gamma-1}$, for $\gamma>1$.''')
    st.markdown('')
    st.markdown('''Results are presented for:''')
    st.markdown(''' - $\gamma=1$ ($\mathsf{sparsemax}$)''')
    st.markdown(' - $\gamma=1.5$ ($1.5$-$\mathsf{entmax}$).''')
with col2:
    st.markdown(''' #### $\mathsf{log\\text{-}margin}$''')
    st.markdown('''Limit case for *softmax*, ($\gamma=1$, *i.e.,* $\delta = +\infty$)''')
    #st.latex('''\\begin{align*}
    #s(x,y) &= \|\\bm{z}_{1:k(y)}-z_{k(y)}\mathbf{1}\|_\infty \\nonumber \\\ \\\ &= z_1 - z_{k(y)} \\nonumber
    #= \log \\frac{p_1}{p_{k(y)}}\\nonumber,
#\end{align*}''')
    st.markdown('''$s(x,y) = \|\\bm{z}_{1:k(y)}-z_{k(y)}\mathbf{1}\|_\infty$''')
    st.markdown('''$\quad\quad\quad= z_1 - z_{k(y)}$''')
    st.markdown('''$\quad\quad\quad= \log \\frac{p_1}{p_{k(y)}}$''')
    st.markdown('''**log-odds ratio** between the most probable class and the true one.''')
    st.markdown('''Calibration of this non-conformity score leads to thresholding the odds ratio $p_1/ p_{k(y)}$''')
with col3:
    st.markdown(''' #### $\mathsf{opt\\text{-}entmax}$''')
    st.markdown('''$\gamma$($1<\gamma<2$) is treated as a hyperparameter, 
                tuned to minimize the average prediction set size.''')
    st.image('images/gamma_entmax.png')
st.markdown('''The choice of score is task-dependent and crutial for the efficiency of a conformal predictor.  
            Check out the 🎯 **Experimental Results** page see the new scores' performance.''')
