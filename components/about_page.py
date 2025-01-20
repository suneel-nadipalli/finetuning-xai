import streamlit as st

def about_page():
    # About Page Content
    st.title("About This Project")
    
    st.markdown("---")

    # Brief Overview
    st.markdown("""
    ### Overview
    This project explores the inner workings of transformer models, specifically the **BERT Base Uncased** model. 
    The main focus is on understanding how fine-tuning affects the model's representations across layers for various tasks.
    The study examines:
    - **Pre-trained and Fine-tuned Models**: Highlighting how task-specific fine-tuning transforms internal activations.
    - **Layer-wise Analysis**: Comparing activations in early (Layer 3), middle (Layer 6), and late (Layer 12) layers.
    - **Task Variability**: Evaluating how the model adapts to distinct datasets and tasks.
    """)

    # Motivation
    st.markdown("""
    ### Motivation
    Transformer models like BERT have revolutionized natural language processing (NLP), 
    but their complexity often makes it challenging to interpret how they work. This project aims to:
    - Provide **visual insights** into token-level activations.
    - Help users understand the evolution of representations during fine-tuning.
    - Enable better intuition about how transformers adapt to specific tasks.
    """)

    # Features
    st.markdown("""
    ### Features
    1. **Layer-wise Visualization**: Examine activations in Layers 3, 6, and 12.
    2. **Pre-trained vs Fine-tuned**: Compare activations between pre-trained and fine-tuned models for each task.
    3. **Dataset-Specific Insights**: Explore how BERT adapts to tasks like sentiment analysis, genre classification, and topic categorization.
    4. **Interactive Visualizations**: View activations as bar graphs or color-coded sentence highlights.
    """)

    # GitHub Repository
    st.markdown("""
    ### GitHub Repository
    The full code for this project, including methods for fine-tuning BERT, extracting activations, and training sparse autoencoders, 
    is available on GitHub. Feel free to explore and contribute!
    """)
    st.markdown("[View on GitHub](https://github.com/suneel-nadipalli/transformer-mi)", unsafe_allow_html=True)

    # Background and Technical Details
    st.markdown("""
    ### Background and Technical Details
    - **Model**: This project uses the `bert-base-uncased` model from the Hugging Face Transformers library.
    - **Tasks**: The model is fine-tuned for:
    - Binary sentiment classification (IMDb).
    - Multi-class genre classification (Spotify).
    - Topic classification (News).
    - **Layers**: Layers 3, 6, and 12 were selected to represent different stages of feature learning:
    - **Layer 3**: Captures general syntactic patterns.
    - **Layer 6**: Transitions to semantic understanding.
    - **Layer 12**: Specializes in task-specific features.
    - **Visualization**: Sparse AutoEncoders (SAEs) are used to isolate and analyze monosemantic features.
    """)

    # Closing Note
    st.markdown("""
    Thank you for exploring this project! I hope it has helped you gain insights into transformer models
    and inspires further exploration into interpretability and representation learning.
    """)
