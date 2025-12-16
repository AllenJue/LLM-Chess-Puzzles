"""
Shared academic/publication style configuration for all graphs.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    """Configure matplotlib and seaborn for AI journal publication style.
    
    Matches standards for NeurIPS, ICML, AAAI, and similar AI conferences:
    - Sans-serif fonts in figures (Arial/Helvetica) for better readability
    - High resolution (300 DPI minimum)
    - Clean, minimal design
    - Colors distinguishable in grayscale
    """
    sns.set_style("whitegrid")
    
    # Typography - sans-serif font for figures (AI journal standard)
    # Many AI journals (NeurIPS, ICML, AAAI) prefer sans-serif in figures
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
    
    # Text sizes (AI journal standard: minimum 10pt, typically 11-12pt)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Line and grid styling
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Figure defaults (AI journal standards: 300 DPI minimum, 600 DPI for line art)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300  # Minimum for images, can increase to 600 for line art
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'none'
    plt.rcParams['savefig.format'] = 'png'  # PNG is widely accepted, can also use PDF/EPS

def apply_academic_axes(ax, remove_top_right=True):
    """Apply AI journal styling to axes.
    
    Features:
    - Removes top/right spines (clean, minimal design)
    - Subtle grid lines for readability
    - Clear axis lines
    """
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Make remaining spines more visible (AI journal standard: clear but not heavy)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Grid styling (subtle, doesn't distract from data)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.6)

def create_academic_legend(ax, handles, labels=None, loc='lower right', fontsize=10):
    """Create a styled legend for AI journal figures.
    
    Args:
        ax: Matplotlib axes object
        handles: List of legend handles (Patch objects, Line2D, etc.)
        labels: Optional list of labels. If None, extracts labels from handles.
        loc: Legend location (default: 'lower right' per AI journal conventions)
        fontsize: Font size for legend (default: 10pt, AI journal minimum)
    
    Returns:
        Matplotlib legend object
    """
    # If labels not provided, extract from handles
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    
    # AI journal style: clear frame, subtle shadow, readable font
    legend = ax.legend(handles, labels, loc=loc, fontsize=fontsize, 
                      frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                      edgecolor='black', facecolor='white')
    return legend

