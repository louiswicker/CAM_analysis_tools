#===============================================================================
# A function to add panel labels for plots.  

def label_panels(axs, panel_labels, fontsize=14, adjust_loc=None):

    from operator import add

    base_loc = [0.025, 0.975]

    if adjust_loc:
        loc = list(map(add, base_loc, adjust_loc)) 
    else:
        loc = base_loc

    if isinstance(axs, list):
        
        for i, ax in enumerate(axs):        
            ax.text(*loc, panel_labels[i], transform=ax.transAxes, fontsize=fontsize, \
                  fontweight='medium', va='top', ha='left')

    else:
        
        for i, ax in enumerate(axs.flatten()):
    # Add the panel label
            ax.text(*loc, panel_labels[i], transform=ax.transAxes, fontsize=fontsize, \
                  fontweight='medium', va='top', ha='left')

        return axs
