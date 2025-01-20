import sys
import streamlit as st

sys.path.append('../..')

from components.dataset_carousel import *

import streamlit_shadcn_ui as ui

def landing_page():
    st.title("Exploring Activations in BERT Models with Fine-Tuning")
    
    st.markdown("---")

    # Brief Description
    st.markdown("""
    ### Welcome!
    This platform offers an interactive exploration of the **BERT Base Uncased** model, 
    focusing on how fine-tuning transforms its internal representations. You can visualize 
    and compare **token-level activations** across:
    - **Pre-trained and Fine-tuned Models**: Understand how fine-tuning affects model representations for different tasks.
    - **Three Layers**: Explore Layer 3 (early), Layer 6 (middle), and Layer 12 (late) to see how features evolve through the network.
    - **Three Datasets**: Choose from IMDb, Spotify, and News datasets to observe task-specific activations.

    By offering tools to dynamically visualize activations, this app provides insights into the 
    inner workings of transformer models and their adaptability to diverse tasks.
    """)

    # How to Use
    st.markdown("""
    ### How to Use:
    1. **Select a Layer**:
    - Choose from Layer 3, 6, or 12 to explore activations.
    - Layers represent different levels of feature abstraction: early layers capture general syntactic patterns, middle layers transition to semantics, and late layers specialize in task-specific features.
                
    2. **Select a Dataset**:
    - Pick from IMDb, Spotify, or News datasets to observe task-specific activations.
    - Each dataset corresponds to a different task and set of classes, influencing the model's internal representations.
    - Use the dataset descriptions below to understand the tasks, classes, and objectives.
                
    3. **Enter a Sentence**:
    - Enter a custom sentence to visualize token-level activations.
    - Each token is highlighted with a color intensity corresponding to its activation value.

    4. **Compare Configurations**:
    - Toggle between pre-trained and fine-tuned versions of the BERT model to observe differences in activation patterns.

    5. **Visualize**:
    - Choose a graph to view activations as a bar chart or a color-coded sentence visualization for an intuitive understanding.
    """)

    st.markdown("---")

    # Dataset Cards Section
    st.subheader("Datasets Overview")
    st.markdown("""
    Below are the datasets used in this experiment. Each card provides details about the dataset, including 
    the task it represents and the associated classes. Feel free to explore the datasets to gain a deeper 
    understanding of their role in fine-tuning the model.
    """)

    sel_ds = ui.tabs(
            options=["IMDb", "Spotify", "News"],
        )

    if sel_ds == "IMDb":
        dataset_card(
            name="IMDb",
            desc="""
            The IMDb dataset contains 50,000 movie reviews for sentiment analysis.
            It is split into 25,000 reviews for training and 25,000 reviews for testing.
            """,
            task="Binary Sentiment Analysis",
            classes=["Positive", "Negative"],
            pt_acc=54.0,
            ft_acc=93.0,
        )

    elif sel_ds == "Spotify":
        dataset_card(
            name="Spotify",
            desc="""
            The Spotify dataset contains 35,000 reviews for the Spotify app.
            This is a multi-class classification task with 5 classes,
            each representing a different rating on the 1-5 scale.
            """,
            task="Multi-Class Sentiment Analysis",
            classes=["1", "2", "3", "4", "5"],
            pt_acc=12.3,
            ft_acc=57.5,
        )

    elif sel_ds == "News":
        dataset_card(
            name="News",
            desc="""
            The News dataset contains 25,000 news articles from various sources.
            This is a multi-class (topc) classification task with 20 classes,
            each representing a different news category. For the purpose of this demo, 5 of the 20 classes were selected.
            """,
            task="Topic Classification",
            classes=["Wellness", "Sports", "Entertainment", "Politics", "Food"],
            pt_acc=12.0,
            ft_acc=92.0,
        )
