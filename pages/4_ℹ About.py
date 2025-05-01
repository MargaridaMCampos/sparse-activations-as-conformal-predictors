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
    

'''
# Paper details

Presented at the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) on May, 2025.  

[Code](https://github.com/deep-spin/sparse-activations-cp) to reproduce experiments is publicly available.
'''
''' #### Authors
'''
col1,col2,col3,col4,col5 = st.columns(5,gap='large')
with col1:
    st.image('images/authors/margarida.jpeg',use_container_width=True)
    st.markdown('[Margarida M. Campos](https://scholar.google.com/citations?user=vORjPgMAAAAJ&hl=en)')
with col2:
    st.image('images/authors/calem.jpeg',use_container_width=True)
    st.markdown('[João Calém](https://scholar.google.com/citations?user=F1bblH4AAAAJ&hl=en)')
with col3:
    st.image('images/authors/sophia.jpeg',use_container_width=True)
    st.markdown('[Sophia Sklaviadis](https://scholar.google.com/citations?user=wkFXjGAAAAAJ&hl=en)')
with col4:
    st.image('images/authors/mario.jpg',use_container_width=True)
    st.markdown('[Mártio A.T. Figuiredo](https://scholar.google.com/citations?user=S-pd0NwAAAAJ&hl=en)')
with col5:
    st.image('images/authors/andre.jpeg',use_container_width=True)
    st.markdown('[André F.T. Martins](https://scholar.google.com/citations?user=mT7ppvwAAAAJ&hl=en)')

''' #### Affiliations
'''

col1,col2,col3,col4 = st.columns(4,gap='large')
with col1:
    st.image('images/institutions/tecnico.png',use_container_width=True)
    #st.markdown('[Instituto Superior Técnico, University of Lisbon](https://tecnico.ulisboa.pt/)')
with col2:
    st.image('images/institutions/it.png',use_container_width=True)
    #st.markdown('[Instituto de Telecomunicações](https://www.it.pt/ITSites/Index/1)')
with col3:
    st.image('images/institutions/ellis.png',use_container_width=True,width=30)
    #st.markdown('[ELLIS Unit Lisbon](https://ellis.eu/units/lisbon)')
with col4:
    st.image('images/institutions/unbabel.png',use_container_width=True)
    #st.markdown('[Unbabel](https://unbabel.com/)')

ncol1,ncol2,ncol3,ncol4 = st.columns(4,gap='large')
with ncol1:
    #st.image('images/institutions/tecnico.png',use_container_width=True)
    st.markdown('[Instituto Superior Técnico, University of Lisbon](https://tecnico.ulisboa.pt/)')
with ncol2:
    #st.image('images/institutions/it.png',use_container_width=True)
    st.markdown('[Instituto de Telecomunicações](https://www.it.pt/ITSites/Index/1)')
with ncol3:
    #st.image('images/institutions/ellis.png',use_container_width=True,width=30)
    st.markdown('[ELLIS Unit Lisbon](https://ellis.eu/units/lisbon)')
with ncol4:
    #st.image('images/institutions/unbabel.png',use_container_width=True)
    st.markdown('[Unbabel](https://unbabel.com/)')
''' #### Citation 
```
@inproceedings{campos2025sparseactivationsconformalpredictors,
  title={Sparse Activations as Conformal Predictors},
  author={Margarida M. Campos and João Calém and Sophia Sklaviadis and M{\'a}rio A. T. Figueiredo and Andr{\'e} F. T. Martins},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2025}
}
```
'''

'''#### Contact  
For any additional information or problem reporting, please contact: <margarida.campos@tecnico.ulisboa.pt>.
'''

'''
'''