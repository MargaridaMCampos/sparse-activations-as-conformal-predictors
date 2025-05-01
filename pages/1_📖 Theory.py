import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import torch
from random import randint
from torch.nn.functional import softmax
from entmax.activations import sparsemax
from entmax.root_finding import entmax_bisect
from confpred import ConformalPredictor,SparseScore,SoftmaxScore, LimitScore
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
    
@st.cache_data
def load_example_data():
    with open('./data/examples_info.pickle', 'rb') as handle:
        return pickle.load(handle)

@st.cache_data
def load_cp_info():
    with open('./data/conformal_prediction_info.pickle', 'rb') as handle:
        return pickle.load(handle)

def compute_quantile(cal_scores, selection, alpha):
    n_cal = cal_scores.shape[0]
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    return np.quantile(cal_scores[selection], q_level, method='higher')

def plot_threshold_histogram(df, selection, threshold):
    fig = px.histogram(df, x=selection, nbins=10)
    fig.add_vline(x=threshold, line_color='green',
              annotation_text=fr"q={threshold:.2f}", 
              annotation_position="top right",
              annotation_font_color="green")
    fig.update_layout(width=300, 
                      height=200,
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(
                        title=dict(
                            text=r"Frequency"
                                )
                            ),
                        xaxis=dict(
                            title=dict(
                                text=r"$s(x,y)$"
                            )
                        ),
                      )
    return fig

def plot_example_scores(example_scores, selection, threshold):
    color_map = {i:el for i,el in enumerate(px.colors.qualitative.Pastel)}
    example_scores['threshold'] = threshold
    example_scores['opacity'] = np.where(example_scores[selection] < threshold, 1, 0.2)
    fig = go.Figure()
    for idx, row in example_scores.iterrows():
        fig.add_trace(go.Bar(
            x=[row['classes']],
            y=[row[selection]],
            marker_color=color_map[idx],
            opacity=row['opacity'],
            name=row['classes'],
            showlegend=False
        ))
    fig.add_hline(y=threshold, line_color='green',
                  annotation_text=r"$\hat{q}$", 
                        annotation_position="top right",
                        annotation_font_color="green",
                        annotation_font_size=16,
                        annotation_font_weight='bold')
    fig.update_layout(width=300, 
                      height=150, 
                      margin =dict(l=0, r=0, b=0, t=0),
                      yaxis = dict(
                          title = dict(text = r'$s(x,y)$')
                      )
                      )
    return fig

examples_info = load_example_data()
cp_info = load_cp_info()

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Conformal Prediction", "\U0001F4C8 Sparse Activations", "\U0001F517 Connecting Both"])
betas = np.sort(np.concat((np.linspace(0.01, 10, 100),np.arange(1,10,dtype=float))))

index_list = list(examples_info['samples'].keys())
    #index = index_list[1]
if "index" not in st.session_state:
    st.session_state.index = index_list[1]  # or whatever default you want
with st.sidebar:
    if st.button('🔄 New Image',type='tertiary'):
        st.session_state.index = index_list[randint(0,len(index_list)-1)]
        
with tab1:  
    st.markdown('#### Set Prediction with Guarantees')
    col1, col2 = st.columns([0.5,0.5],gap = 'large')
    with col1:
        st.markdown('''You get:''')
        st.markdown(''' - a conformal predictor that outputs **prediction sets** 
                    guaranteed to contain the ground truth with a user-chosen confidence level:''')
        st.latex('''\mathbb{P}\\big(Y_\\text{test}\in \mathcal{C}_\\alpha(X_\\text{test})\\big)\geq 1- \\alpha''')
        st.markdown(''' - model-agnostic and distribution-free framework''')
        st.markdown('''You need:''')
        st.markdown(''' - a choice of non-conformity score, $s$ (a measure of how *unlikely*
                    an observation is''')
        st.markdown(''' - *exchangeable* calibration data''')
    with col2:
        option_map = {
            'softmax': "InvProb",
            'sparsemax': "🆕 sparsemax",
            'entmax': "🆕 1.5-entmax",
            'limit': '🆕 log-margin'
        }
    
        selection = st.radio(
            "Non-conformity score (**s**)",
            options=option_map.keys(),
            help='''
             - $\\bm{z}=f(\\bm{x})$: model predictions assumed to be sorted in descending order, $z_1 \ge z_2 \ge ... \ge z_K$
             - $k(y)$: the index of label $y$ in the sorted array $\\bm{z}$
            ''',
            format_func=lambda option: option_map[option],
            horizontal=False,
            captions=[r'$s(x,y)=1-\text{softmax}_y(x)$',
                      r'$s(x,y) = \sum_{k=1}^{k(y)-1} (z_k - z_{k(y)})$',
                      r'$s(x,y) = \|\bm{z}_{1:k(y)}-z_{k(y)}\mathbf{1}\|_2$',
                      r'$s(x,y) = \|\bm{z}_{1:k(y)}-z_{k(y)}\mathbf{1}\|_\infty = z_1 - z_{k(y)}= \log \frac{p_1}{p_{k(y)}}$']
            )
        alpha = 1 - st.slider(r'Confidence Level (**$1-\alpha$**)', 
                                min_value=0.9, 
                                max_value=0.99)
    
    with st.sidebar:
        
        col1,col2,col3 = st.columns([0.3,0.5,0.3])
        with col2:
            #st.write("<b style='text-align: center; color: grey;'>Sample</b>", unsafe_allow_html=True)
            index = st.session_state.index
            fig, ax = plt.subplots()
            ax.imshow(examples_info['samples'][index]['example'])
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig, use_container_width=False)
            st.markdown(r"$y_\text{true}$" +f": **{examples_info['classes'][examples_info['samples'][index]['example_label'].item()]}**")
            

    threshold = compute_quantile(cp_info['cal_scores'], selection, alpha)
    
    #martelada -> FIX
    n_cal = cp_info['cal_scores'].shape[0]
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    
    col1, col2 = st.columns([0.5,0.5], gap='medium',border=True)

        
        #st.write(fr"**z** =  {examples_info['samples'][index]['example_logits'].round(1)}")
    with col1:
        st.markdown('''#### Calibration ''')
        st.markdown(r"1. Compute scores, $s$, for calibration points")
        st.markdown(r"2. Obtain $\frac{\lceil(n_\text{cal}+1)(1-\alpha)\rceil}{n_\text{cal}}$ empirical quantile")

        hist = plot_threshold_histogram(cp_info['cal_scores'], selection, threshold)
        st.plotly_chart(hist, use_container_width=True,config = config)
        st.markdown(f'''**{(100*q_level):.2f}%** of calibration points are less non-conformal than '''+
                    '''$\hat{q}$='''+f'''{threshold:.2f}.''')
        
        st.markdown('''Labels yielding '''+r'$s(x,y)<\hat{q}$'+
                    f' will belong to the prediction set.')

    with col2:
        st.markdown('#### Prediction')
        st.markdown('1. Obtain model predictions **$z$**')
        color_map = {i:el for i,el in enumerate(px.colors.qualitative.Pastel)}
        logits_df = pd.DataFrame({'classes':examples_info['classes'],
                                   'logits':examples_info['samples'][index]['example_logits']})
        fig = go.Figure()
        for idx, row in logits_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['classes']],
                y=[row['logits']],
                marker_color=color_map[idx],
                name=row['classes'],
                showlegend=False
            ))
        fig.update_layout(width=300, 
                          height=100,
                          margin =dict(l=0, r=0, b=0, t=0),
                          xaxis = dict(showticklabels=False),
                          yaxis = dict(
                            title = dict(text = r'$z_i$')
                        ))
        st.plotly_chart(fig,use_container_width=True, config=config)
        st.markdown(r'2. Compute $s(x_\text{test},y)$ for all possible labels')
        scores_plot = plot_example_scores(examples_info['samples'][index]['example_scores'], selection, threshold)
        st.plotly_chart(scores_plot, use_container_width=True,config = config)
        prediction_set = list(np.array(examples_info['classes'])[examples_info['samples'][index]['example_scores'][selection] < threshold])
        st.markdown(r'Prediction set: {'+f'**{",".join(str(i) for i in prediction_set)}**' + r'}')
    st.markdown('')
with tab2:
    #st.subheader("Sparse Alternatives to Softmax")
    #expander = st.expander("Theoretical Details")
    #expander.write('''...''')

    col1,col2 = st.columns([0.6,0.4], gap='large')
    
    #ADD LINKS!!
    with col1:
        st.write('#### The $\gamma$-entmax family')
        st.markdown(''' - ***softmax*** - typical activation used to map a classifier's predictions
                    into probabilities over classes''')
        st.markdown(''' - ***sparsemax*** ([Martins and Astudillo (2016)](www.)) 
                    - alternative activation capable of assigning
                    zero probabilities''')
        st.markdown(''' - **$\mathbf{\gamma}$-entmax** - a family of transformations [Peters et al. (2019)]()
                    that has *sparsemax* ($\gamma=2$) and *softmax* ($\gamma=1$) as particular cases.  
                    For $\gamma>1$, outputs can be sparse.''')
        gamma_selection = st.slider(r'$\gamma$', 
                                    min_value=1.1, 
                                    max_value=4.0,
                                    value = 1.5)
    with col2:
        
        space = np.linspace(-3, 3, 50)
        t = torch.tensor(np.column_stack([space, np.zeros_like(space)]))
        activations = pd.DataFrame({
            't': space,
            'softmax': softmax(t, dim=-1)[:, 0].numpy(),
            'sparsemax': sparsemax(t, dim=-1)[:, 0].numpy(),
            fr'{gamma_selection}-entmax': entmax_bisect(t, dim=-1, alpha=gamma_selection)[:, 0].numpy()
        }).melt(id_vars=['t'], var_name='activation', value_name='value')
    
        fig = px.line(activations, x='t', y='value', color='activation')
        fig.update_layout(legend=dict(orientation='h', y=-0.2), template='plotly_dark')
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            height=370,
            yaxis=dict(
                title=dict(
                    text=r"$\gamma\text{-entmax}(z)_1$"
                )
            ),
            xaxis=dict(
                title=dict(
                    text=r"$z_1\quad(z=[z_1,0])$"
                )
            ),
            legend=dict(
                    orientation='v',
                    yanchor="bottom",
                    y=0.1,
                    xanchor="right",
                    x=1,
                title=dict(
                    text=""
                ))
            )
        
        
        st.plotly_chart(fig,use_container_width=True,config = config)
    
    st.write('#### Adjusting Sparsity')
    st.markdown('''**Temperature scaling** - $f(\\beta\mathbf{z})$ - the calibration technique allowing to adjust
                how peaked a transformation is, allows for the **control of the sparsity** of
                $\gamma$-entmax outputs''')
    
    beta_slider = st.slider(r'$\beta^{-1}$', min_value=0.0, max_value=10.0, step=1.0,value=1.0)

    z = torch.tensor([1, -1, -0.3, 0.3, 0.01])
    soft, sparse, ent = np.empty((5, len(betas))), np.empty((5, len(betas))), np.empty((5, len(betas)))
    for idx, beta in enumerate(betas):
        soft[:, idx] = softmax((1 / beta) * z, dim=0).numpy()
        sparse[:, idx] = sparsemax((1 / beta) * z).numpy()
        ent[:, idx] = entmax_bisect((1 / beta) * z, alpha=gamma_selection).numpy()

    def plot_beta(df, transformation):
        df_subset = df[df['beta'].round() == round(beta_slider)]
        df_subset['supported_beta'] = df_subset['prob'] > 0
        fig = px.line(df, x='beta', y='prob', color='class', title=transformation)
        fig.add_vline(x=beta_slider, line_color='red')
        fig.update_layout(template='plotly_dark')
        return fig

    df_soft = pd.DataFrame(soft.T, columns=range(5)).assign(beta=betas)#.melt('beta', var_name='class', value_name='prob').assign(transformation='softmax')
    df_sparse = pd.DataFrame(sparse.T, columns=range(5)).assign(beta=betas)#.melt('beta', var_name='class', value_name='prob').assign(transformation='sparsemax')
    df_ent = pd.DataFrame(ent.T, columns=range(5)).assign(beta=betas)#.melt('beta', var_name='class', value_name='prob').assign(transformation='entmax')
    
    col1, col2, col3 = st.columns(3,gap='large')
    colors = ["#2d7dd2","#97cc04","#eeb902","#f45d01","#474647"]
    with col1: 
        fig = go.Figure()
        for i in range(0, 5):
            line_size = 3 if df_soft[df_soft['beta'] == beta_slider][i].max() else 1.5
            fig.add_trace(go.Scatter(x=df_soft['beta'], y = df_soft[i], mode='lines',
                name=fr"$p_{i}$",
                line=dict(color=colors[i], width=line_size),
                connectgaps=True,
                showlegend=False
            ))
        fig.add_vline(x=beta_slider, line_color='red')
        fig.update_layout(
                        margin=dict(l=0, r=0, b=50, t=50),
            height=350,
            title=dict(
                text=r"$\text{Softmax}$",
            ),
            yaxis=dict(
                title=dict(
                    text=r"$\gamma\text{-entmax}(\beta z)$"
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True,config = config)
        #st.plotly_chart(plot_beta(df_soft, 'Softmax'), use_container_width=True)
    with col2: 
        fig = go.Figure()
        for i in range(0, 5):
            line_size = 3 if df_ent[df_ent['beta'] == beta_slider][i].max() else 1.5
            fig.add_trace(go.Scatter(x=df_ent['beta'], y = df_ent[i], mode='lines',
                name=fr"$p_{i}$",
                line=dict(color=colors[i], width=line_size),
                connectgaps=True,
                showlegend=False
            ))
        fig.add_vline(x=beta_slider, line_color='red')
        fig.update_layout(
            margin=dict(l=0, r=0, b=50, t=50),
            height=350,
            title=dict(
                text=r"$\gamma\text{-entmax}$",
            ),
            xaxis=dict(
                title=dict(
                    text=r"$\beta^{-1}$"
                )
            )
        )

        st.plotly_chart(fig, use_container_width=True,config = config)
        #st.plotly_chart(plot_beta(df_ent, 'Entmax'), use_container_width=True)
    with col3: 
        fig = go.Figure()
        for i in range(0, 5):
            line_size = 3 if df_sparse[df_sparse['beta'] == beta_slider][i].max() else 1.5
            fig.add_trace(go.Scatter(x=df_sparse['beta'], y = df_sparse[i], mode='lines',
                name=fr"$p_{i}$",
                line=dict(color=colors[i], width=line_size),
                connectgaps=True,
            ))
        fig.add_vline(x=beta_slider, line_color='red')
        fig.update_layout(
            margin=dict(l=0, r=0, b=50, t=50),
            height=350,
            title=dict(
                text=r"$\text{Sparsemax}$",
            )
        )
        st.plotly_chart(fig, use_container_width=True,config = config)
        #st.plotly_chart(plot_beta(df_sparse, 'Sparsemax'), use_container_width=True)
    st.markdown('''Applying different activations to prediction logits $[1, -1, -0.3, 0.3, 0.01]$ - see how
                the probabilities for each class change as a function of inverse temperature $\\beta$.  
                Check how $\\beta$ affects the support (classes with non-zero probability) of the output - highglighted in bold.''')
    st.markdown('')
with tab3:
    st.markdown(''' ### The Equivalence''')
    st.markdown('''  
                Our paper shows that there exists a choice of non-conformity score that makes
                conformal prediction equivalent to the temperature scaling of the $\gamma$-entmax transformation. 
                ''')
    
    container1 = st.container(border=True)
    with container1:
        st.markdown('''#### Calibration''')
        col1,col2 = st.columns(2, gap='large')
        with col1:
            
            alpha_2 = 1 - st.slider(r'Confidence Level ($1-\alpha$) ', min_value=0.9, max_value=0.99)
            gamma_selection_2 = st.slider(r'$\gamma$', 
                                        min_value=1.1, 
                                        max_value=2.0,
                                        value = 1.5)
        with col2:
            score = SparseScore(gamma_selection_2)
            cal_scores = score.get_single_score(examples_info['cal_true_enc'],
                                                examples_info['cal_proba'])
            n_cal = cal_scores.shape[0]
            q_level_2 = np.ceil((n_cal + 1) * (1 - alpha_2)) / n_cal
            threshold_2 = np.quantile(cal_scores, q_level_2, method='higher')
            cal_scores_df = pd.DataFrame({'entmax':cal_scores})
            hist_2 = plot_threshold_histogram(cal_scores_df,'entmax', threshold_2)
            hist_2.update_layout(width=300, 
                        height=200,
                        margin =dict(l=0, r=0, b=0, t=0),
                        yaxis = dict(
                            title = None
                        )
                        )
            st.plotly_chart(hist_2, use_container_width=True,key=selection,config=config)
    container2 = st.container(border=True)
    with container2:
        st.markdown('''#### Prediction''')
    
        col1, col2 = st.columns(2, gap='large')
                
        with col1:
            def get_example_scores_df(index):
                example_scores = {'classes':examples_info['classes']}
                cp = ConformalPredictor(score)
                cp.calibrate(examples_info['cal_true_enc'], 
                            examples_info['cal_proba'], 0.9)
                cp.predict(np.expand_dims(examples_info['samples'][index]['example_logits'],axis=0))
                example_scores[f'entmax'] = cp.test_scores[0]
                return pd.DataFrame(example_scores)
            test_scores_df = get_example_scores_df(index)
            scores_plot = plot_example_scores(test_scores_df, 'entmax', threshold_2)
            prediction_set_2 = list(np.array(test_scores_df[test_scores_df['entmax'] < threshold_2]['classes']))
            scores_plot.update_layout(
                        height=200,
                        margin =dict(l=0, r=0, b=0, t=0)
                        )
            st.plotly_chart(scores_plot, key=str(selection)+str(alpha), use_container_width=True, config=config)
            st.markdown('$C_{\\alpha}(x)$ = {'+f'**{",".join(str(i) for i in prediction_set_2)}**' + r'}')

        
        with col2:
            sample_logits = torch.tensor(examples_info['samples'][index]['example_logits'])
            ent_sample = np.empty((len(sample_logits), len(betas)))
            for idx, beta in enumerate(betas):
                ent_sample[:, idx] = entmax_bisect((1 / beta) * sample_logits, alpha=gamma_selection_2).numpy()
            
            df_ent_sample = pd.DataFrame(ent_sample.T, columns=range(len(sample_logits))).assign(beta=betas)#.melt('beta', var_name='class', value_name='prob').assign(transformation='entmax')
            delta = 1 / (gamma_selection_2 - 1)
            beta_calc = threshold_2 / delta
            color_map = {i:el for i,el in enumerate(px.colors.qualitative.Pastel)}
            fig = go.Figure()
            for i in range(0, len(sample_logits)):
                line_size = 3 if df_ent_sample[df_ent_sample['beta'].round() == round(beta_calc)][i].max() else 1.5
                fig.add_trace(go.Scatter(x=df_ent_sample['beta'], y = df_ent_sample[i], mode='lines',
                    name=fr"$p_{i}$",
                    line=dict(color = color_map[i],
                            width=line_size), #color=colors_sample[i],
                    connectgaps=True,
                ))
            fig.add_vline(x=beta_calc, line_color='green',
                        annotation_text=r"$\beta = \frac{1}{\hat{q}(\gamma-1)}$", 
                        annotation_position="top right",
                        annotation_font_color="green",
                        annotation_font_size=16,
                        annotation_font_weight='bold')
            fig.update_layout(width=300, 
                        height=200, 
                        margin =dict(l=0, r=0, b=0, t=0),
                        yaxis = dict(
                            title = dict(text = r"$\gamma\text{-entmax}(\beta z)$")
                        ),
                        xaxis = dict(
                            title = dict(text = r"$\beta$")
                        )
                        )
            st.plotly_chart(fig, use_container_width=True,config = config)
            st.markdown('$S(\\beta\mathbf{z};\gamma)$ =  {'+f'**{",".join(str(i) for i in prediction_set_2)}**' + r'}')

        st.markdown('''The prediction set from the conformal predictor, $C_{\\alpha}(x)$,
                    corresponds to the support of $\gamma\\text{-entmax}$, $S(\\beta\mathbf{z};\gamma)$, 
                    for temperature $\\beta^{-1} = \hat{q}(\gamma - 1)$.''')
        st.markdown('')