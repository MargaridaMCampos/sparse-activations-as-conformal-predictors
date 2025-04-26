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

st.sidebar.header("place")

@st.cache_data
def load_cp_info():
    with open('./data/conformal_prediction_info.pickle', 'rb') as handle:
        return pickle.load(handle)

def compute_quantile(cal_scores, selection, alpha):
    n_cal = cal_scores.shape[0]
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    return np.quantile(cal_scores[selection], q_level, method='higher')

def plot_threshold_histogram(df, selection, threshold):
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f'{selection}', bin=True),
        y='count()'
    ).properties(width=300, height=300)
    rule = alt.Chart(pd.DataFrame([{"threshold": threshold}])).mark_rule(color='red').encode(
        x='threshold:Q'
    )
    return chart + rule

cp_info = load_cp_info()

def plot_example_scores(example_scores, selection, threshold):
    example_scores['threshold'] = threshold
    bar = alt.Chart(example_scores).mark_bar().encode(
        x=alt.X('classes:N'),
        y=f'{selection}',
        color=alt.Color('classes:N',
                            scale = alt.Scale(domain = cp_info['classes'],
                                              range = ['#81323E',
                                                        '#AB713B',
                                                        '#BACA4E',
                                                        '#88DA71',
                                                        '#96E8BA',
                                                        '#328173',
                                                        '#3B75AB',
                                                        '#5F4ECA',
                                                        '#C471DA',
                                                        '#E896C5']),
                            legend = None),
        opacity= alt.condition(getattr(alt.datum, selection) < alt.datum.threshold,
            alt.value(1), alt.value(0.2)),
        
    key='classes'
    ).properties(width=300, height=300)
    rule = alt.Chart(pd.DataFrame([{"threshold": threshold}])).mark_rule(color='red').encode(
        y='threshold:Q'
    ).properties(width=300, height=300)
    return bar + rule

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------



tab1, tab2, tab3 = st.tabs(["📊 Conformal Prediction", "📈 Sparse Activations", "🔗 Equivalence"])
betas = np.linspace(0.01, 10, 100)

# -----------------------------------------------------------------------------
# Tab 1: Conformal Prediction
# -----------------------------------------------------------------------------
with st.sidebar:
    alpha = 1 - st.slider(r'Confidence Level ($1-\alpha$)', min_value=0.9, max_value=0.99)
with tab1:
    option_map = {
        'softmax': "InvProb",
        'sparsemax': "Sparsemax",
        'entmax': "1.5-entmax",
        'limit': 'log-margin'
    }
    selection = st.pills(
        "Non-conformity score",
        help='**sjs**',
        default='softmax',
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single"
    )

    threshold = compute_quantile(cp_info['cal_scores'], selection, alpha)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(cp_info['example'])
        ax.set_xticks([])  # <-- these two lines were the issue
        ax.set_yticks([])
        st.pyplot(fig)
    with col2:
        st.altair_chart(plot_threshold_histogram(cp_info['cal_scores'], selection, threshold),use_container_width=True)
    with col3:
        st.altair_chart(plot_example_scores(cp_info['example_probs'], selection, threshold),use_container_width=True)
# -----------------------------------------------------------------------------
# Tab 2: Sparse Activations
# -----------------------------------------------------------------------------

with tab2:
    st.subheader("Sparse Alternatives to Softmax")
    expander = st.expander("Theoretical Details")
    expander.write('''...''')  # Omitted text unchanged

    gamma_selection = st.slider(r'$\gamma$', min_value=1.1, max_value=4.0)

    # Activation Function Plot
    space = np.linspace(-3, 3, 50)
    t = torch.tensor(np.column_stack([space, np.zeros_like(space)]))
    activations = pd.DataFrame({
        't': space,
        'softmax': softmax(t, dim=-1)[:, 0].numpy(),
        'sparsemax': sparsemax(t, dim=-1)[:, 0].numpy(),
        'entmax': entmax_bisect(t, dim=-1, alpha=gamma_selection)[:, 0].numpy()
    }).melt(id_vars=['t'], var_name='activation', value_name='value')

    transform_plot = alt.Chart(activations).mark_line().encode(
        x='t', y='value',
        color=alt.Color('activation:O', legend=alt.Legend(orient='bottom', direction='horizontal'))
    ).properties(title='Activation Functions')
    st.altair_chart(transform_plot,use_container_width=True)

    # Beta Sweeps
    z = torch.tensor([1, -1, -0.3, 0.3, 0.01])
    soft, sparse, ent = np.empty((5, len(betas))), np.empty((5, len(betas))), np.empty((5, len(betas)))
    for idx, beta in enumerate(betas):
        soft[:, idx] = softmax((1 / beta) * z, dim=0).numpy()
        sparse[:, idx] = sparsemax((1 / beta) * z).numpy()
        ent[:, idx] = entmax_bisect((1 / beta) * z, alpha=gamma_selection).numpy()

    beta_slider = st.slider(r'$\beta$', min_value=0.0, max_value=10.0, step=1.0)
    beta_df = pd.DataFrame([{'beta': beta_slider}])

    # Reusable plotting function
    def plot_beta(df, transformation):
        df_subset = df[df['beta'].round() == round(beta_slider)]
        df_subset['supported_beta'] = df_subset['prob'] > 0
        merged_df = df.merge(df_subset[['class', 'supported_beta']], on='class', how='left')
        chart = alt.Chart(merged_df).mark_line().encode(
            x='beta', y='prob', 
            color=alt.Color('class:O',
                   scale=alt.Scale(
            domain=[0,1,2,3,4],
            range=['red', 'green','blue','orenge','black'])),
            size=alt.condition(alt.datum.supported_beta, alt.value(3), alt.value(1))
        ).properties(title=transformation)
        return chart + alt.Chart(beta_df).mark_rule(color='red').encode(x='beta:Q')

    df_soft = pd.DataFrame(soft.T, columns=range(5)).assign(beta=betas).melt('beta', var_name='class', value_name='prob').assign(transformation='softmax')
    df_sparse = pd.DataFrame(sparse.T, columns=range(5)).assign(beta=betas).melt('beta', var_name='class', value_name='prob').assign(transformation='sparsemax')
    df_ent = pd.DataFrame(ent.T, columns=range(5)).assign(beta=betas).melt('beta', var_name='class', value_name='prob').assign(transformation='entmax')

    col1, col2, col3 = st.columns(3)
    with col1: st.altair_chart(plot_beta(df_soft, 'Softmax'),use_container_width=True)
    with col2: st.altair_chart(plot_beta(df_ent, 'Entmax'),use_container_width=True)
    with col3: st.altair_chart(plot_beta(df_sparse, 'Sparsemax'),use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3: Equivalence
# -----------------------------------------------------------------------------

with tab3:
    gamma_selection_2 = st.slider(r'$\gamma$', min_value=1.5, max_value=2.0, step=0.5)
    alpha_2 = alpha
    temp = 'entmax' if gamma_selection_2 == 1.5 else 'sparsemax'
    threshold_2 = compute_quantile(cp_info['cal_scores'], temp, alpha_2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.altair_chart(plot_threshold_histogram(cp_info['cal_scores'], temp, threshold_2),use_container_width=True)
    with col2:
        st.altair_chart(plot_example_scores(cp_info['example_probs'], temp, threshold_2),use_container_width=True)
    with col3:
        z = torch.tensor(cp_info['example_logits'])
        ent = np.empty((len(z), len(betas)))
        for idx, beta in enumerate(betas):
            ent[:, idx] = entmax_bisect((1 / beta) * z, alpha=gamma_selection_2).numpy()
        
        df_ent = pd.DataFrame(ent.T, columns=range(len(z))).assign(beta=betas).melt('beta', var_name='class', value_name='prob')
        delta = 1 / (gamma_selection_2 - 1)
        beta_calc = threshold_2/delta   # <-- this moves with confidence level!

        df_ent_subset = df_ent[df_ent['beta'] <= beta_calc]
        df_ent_subset['supported_beta'] = df_ent_subset['prob'] > 0
        df_ent_subset = df_ent_subset[['class', 'supported_beta']].groupby('class').max().reset_index()
        merged_df = df_ent.merge(df_ent_subset[['class', 'supported_beta']], on='class', how='left')

        merged_plot = alt.Chart(merged_df).mark_line().encode(
            x='beta',
            y='prob',
            color=alt.Color('class:O',
                            scale = alt.Scale(domain = list(range(10)),
                                              range = ['#81323E',
                                                        '#AB713B',
                                                        '#BACA4E',
                                                        '#88DA71',
                                                        '#96E8BA',
                                                        '#328173',
                                                        '#3B75AB',
                                                        '#5F4ECA',
                                                        '#C471DA',
                                                        '#E896C5']),
                            legend=alt.Legend(orient='bottom', direction='horizontal')),
            size=alt.condition(alt.datum.supported_beta, alt.value(3), alt.value(1))
        ).properties(title='Entmax')

        merged_beta_rule = alt.Chart(pd.DataFrame([{'beta': beta_calc}])).mark_rule(color='red').encode(x='beta:Q')

        st.altair_chart(merged_plot + merged_beta_rule,use_container_width=True)


