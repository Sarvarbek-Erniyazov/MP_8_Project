import matplotlib.pyplot as plt
import os
from src.logger import logging_instance

def save_figure(fig_name, folder="reports/figures"):
    
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{fig_name}.png")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        logging_instance.info(f"Grafik saqlandi: {path}")
    except Exception as e:
        logging_instance.error(f"Grafikni saqlashda xato: {e}")