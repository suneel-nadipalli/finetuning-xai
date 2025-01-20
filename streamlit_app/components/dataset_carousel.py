import sys

sys.path.append('../..')

from scripts.sae.train_sae import *
from scripts.sae.infer_sae import *

from components.utils import *
from streamlit_elements import elements, mui

def dataset_card(name, desc, task, classes, pt_acc, ft_acc):

    with elements(f"{name.lower()}_dataset_card"):
        
        with mui.Paper(elevation=3, style={
            "padding": "20px", "margin": "10px",
            "height": "40%",
            "borderRadius": "8px", "backgroundColor": "#f9f9f9"}):
            
            mui.Typography(name, variant="h4", style={"color": "#2b6cb0", "fontWeight": "bold"})

            mui.Typography(desc, variant="body1", style={"marginTop": "10px"},)

            mui.Chip(label=task, color="primary",
                     style={"margin": "10px 0", "fontSize": "0.9rem"})

            mui.Typography(
                "Classes:",
                variant="h6",
                style={"marginBottom": "3px", "color": "#333", "fontWeight": "bold"},
            )

            # Tabs for classes (non-clickable)
            mui.Box(
                sx={"width": "100%", "borderBottom": "1px solid #ddd", "marginBottom": "20px"}
            )(
                mui.Tabs(
                    value=False,  # Disable tab selection
                    onChange=None,  # Remove the event handler
                    variant="scrollable",
                    scrollButtons="auto",
                    allowScrollButtonsMobile=True,
                )(
                    [mui.Tab(label=cls, disabled=True) for cls in classes]  # Disable all tabs
                )
            )

            mui.Box(
            sx={
                "display": "flex",
                "gap": "10px",
                "alignItems": "center",
                "marginTop": "10px",
            }
            )(
                mui.Typography(f"Pre-Trained Acc: {pt_acc}%", variant="body1", color="textSecondary"),
                mui.Typography(f"Fine-Tuned Acc: {ft_acc}%", variant="body1", style={"fontWeight": "bold", "color": "#2ca02c"}),
            )
