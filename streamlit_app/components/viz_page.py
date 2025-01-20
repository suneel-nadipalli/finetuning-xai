import sys, torch
import streamlit as st

sys.path.append('../..')

from scripts.sae.train_sae import *
from scripts.sae.infer_sae import *

from components.utils import *

def viz_page():
    
    st.title("Visualize Token-Level Activations")
    
    st.markdown("""
    ### Explore Transformer Layers
    Use this page to explore how different layers of the **BERT Base Uncased** model capture and learn various features across different tasks. 
    You can examine token-level activations for sentences and compare how representations evolve across:
    - **Layers**: Choose from Layer 3, Layer 6, or Layer 12 to analyze different levels of abstraction.
    - **Datasets**: Select from IMDb, Spotify, or News datasets.
    - **Configurations**: Switch between pre-trained and fine-tuned models to see task-specific adaptations.

    Get started by selecting a dataset, layer, and visualization mode below!
    """)

    config_col, _, viz_col = st.columns([5, 1, 6])

    with config_col:
        
        ds_col, _, layer_col = st.columns([5, 1, 5])

        with ds_col:
            dataset = st.selectbox("Dataset", ["IMDb", "Spotify", "News"])
        
        with layer_col:
            layer = st.select_slider("Layer", ["Layer 3", "Layer 6", "Layer 12",])

        sentence = st.text_area("Sentence", "I love this movie!")

        ft = st.toggle("Fine-Tuned Model", False)


        # Add a legend for the activation values
        st.markdown("""
        #### Activation Value Legend
        The color intensity of the tokens corresponds to their activation values. Use the legend below as a reference:
        """)

        # Add the gradient bar using CSS
        st.markdown("""
        <div style="
            display: flex; 
            align-items: center; 
            gap: 10px; 
            margin-top: 10px;
        ">
            <span style="font-size: 14px; color: #1f77b4;">Low</span>
            <div style="
                flex: 1;
                height: 20px; 
                background: linear-gradient(to right, blue, red); 
                border-radius: 5px;
            "></div>
            <span style="font-size: 14px; color: #d62728;">High</span>
        </div>
        """, unsafe_allow_html=True)
    
    with viz_col:
        if st.button("Visualize"):

            # Token-level activations
            act_dict = activation_helper(dataset, layer, ft, sentence)

            # Extract activation values and normalize them
            activations = [d["Activation"] for d in act_dict]
            normalized_activations = normalize_activations(activations)

            # Generate color-coded sentence

            color, graph = st.tabs(["Color-Coded Sentence", "Activation Graph"])

            with color:
                st.markdown("### Token-Level Activations")
                token_html = " ".join(
                    [
                        f"<span style='background-color: {activation_to_color(norm_act)}; padding: 2px 5px; margin: 2px; border-radius: 3px;'>{d['Token']}</span>"
                        for d, norm_act in zip(act_dict, normalized_activations)
                    ]
                )
                st.markdown(token_html, unsafe_allow_html=True)
            with graph:
                plot_token_activations(act_dict)
