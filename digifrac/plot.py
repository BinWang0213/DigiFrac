import numpy as np
import os

def plot_frac_img(data=None,region=None,hsize=1.0,
                        title=['Frac top depth(mm)','Frac bottom depth(mm)','Frac aperature(mm)'], 
                        colormap='rainbow'):
    #Plot fracture pattern
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.4)

    frac_info=[data]


    plt_region = np.array([region[0],region[1], region[2],region[3]])


    for i in range(len(frac_info)):
        ax = fig.add_subplot(1, len(frac_info), i+1)
        #Plot the field
        region_pix=np.array(plt_region/hsize,dtype=np.int_)
        im_data = frac_info[i][region_pix[2]:region_pix[3],region_pix[0]:region_pix[1]]

        cm = plt.cm.get_cmap(colormap, 10)

        im = ax.imshow(im_data, cmap =cm, 
                            extent = plt_region, 
                            vmin=np.min(frac_info[i]), 
                            vmax=np.max(frac_info[i]),
                            interpolation ='nearest',origin='lower') 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ticks_level = np.linspace(np.min(frac_info[i]),np.max(frac_info[i]),5)
        cbar_num_format = FuncFormatter(lambda x, pos: '{:.1f}'.format(x))

        cbar = fig.colorbar(im, pad=5, ticks=ticks_level,format = cbar_num_format, cax=cax, orientation='vertical')
        ax.set_aspect('equal')
        ax.set_xlabel("x(mm)")
        ax.set_ylabel("y(mm)")
        ax.set_title(title[i])
    
    return fig


