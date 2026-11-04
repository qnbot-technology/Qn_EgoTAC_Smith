#!/usr/bin/env python3
"""Small diagnostic plot; may run in an existing separate plotting environment."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dest=Path(sys.argv[1]);name=sys.argv[2]
with np.load(dest/'region-traces.npz') as z:
    fig,axes=plt.subplots(2,2,figsize=(16,7),sharex=True,sharey=True)
    colors=plt.get_cmap('tab10').colors
    for i,side in enumerate(('left','right')):
        for col,kind in enumerate(('raw','residual')):
            for j,region in enumerate(['thumb','index','middle','ring','pinky','palm']):
                axes[i,col].plot(z['sample_time_s'],z[side+'_'+kind+'_regions'][:,j],label=region,color=colors[j],lw=.9)
            axes[i,col].set_title(side+' / '+('raw region maximum' if col==0 else 'fixed-baseline + deadband residual'))
            axes[i,col].set_ylim(-3,255);axes[i,col].set_ylabel('counts (not force)');axes[i,col].grid(alpha=.2)
    axes[0,0].legend(ncol=6,fontsize=8);axes[1,0].set_xlabel('source sample time (s)');axes[1,1].set_xlabel('source sample time (s)')
    fig.suptitle(name+' | estimated baseline, no unloaded interval confirmed')
    fig.tight_layout();fig.savefig(dest/'raw-vs-residual-traces.png',dpi=130);plt.close(fig)
