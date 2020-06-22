
# imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
# new, updated

def plot_spots_on_image(adata_obj, # adata object, can be from raw_adatas or filtered_adatas
                        metric, # one of 3 kinds of variables:
                               # 1) column in adata_obj.obs table to plot - 'n_counts', 'n_genes', 'percent_mito', 'Cluster', etc.
                               # 2) gene from adata_obj.var_names list for marker gene display
                               # important: this displays genes from adata_obj.raw.X
                               # 3) None for just basic dots
                               
                        res = 'high_10X', # resolution of the image to plot, either 'high_10X' - the high res image that 10X provide 
                                          # or 'highest' - highest resolution, 20x jpeg import from NDPI
                        spotfilt = None, # dictionary of the following type spotfilt={'row_min': 26, 'row_max': 31, 'col_min': 74, 'col_max': 85}
                                        # filter of which spots to plot, if less than all spots then it zooms in
                        transparency_coeff = 1, # transparency coefficient (from 1 to 0) alpha in plotting functions
                        color_map = 'OrRd', # colormap for the continuous values passed in 'metric' variable
                        color_map_clusters = plt.cm.jet, # if metric is discrete ('Cluster' or 'clusters' (from louvain)) then discrete colormap (jet) will be used - easier for cluster differentiation visually
                        spots_display = True, # to display the spots or not
                        spot_color = 0, # color of spots to display if passed metric is None
                        figure_size = [25,25], # size of the figure to display
                        col_max = None, # force boundaries of continuous color map
                        col_min = None, # force boundaries of continuous color map
                        save_pdf_name = None, # if None, doesn't save the figure, if a string, treats that string as full path to file
                        metric_type = None, # type of variable to plot - 'continuous' or 'discrete' - for colormap display choice
                        filter_factor = None, # if you want to plot only certain spots, subset of the adata_obj.obs table
                        highest_res_images_dict = None, # dictionary of highest resolution images (exported from NDPI)
                        images_dict_10x = None, # dictionary of high resolution images provided by 10X Genomics
                        scalefactors_dict = None # dictionary of scale factors for spot coordinates
                                ):
        
    ID = adata_obj.obs['sample_ID'][0]
    
    # checking input resolution
    if res == 'highest':
        image_dict_to_use = highest_res_images_dict
        
        # columns of adata_obj.obs to take spot coordinates from 
        col_x = 'x_highest_adj'
        col_y = 'y_highest_adj'
        
        # radius of dots to plot, in pixels
        # x or y scaling factor?
        dot_radius = scalefactors_dict[ID]['10X_hi_to_20x_highest_y_coord']*scalefactors_dict[ID]['spot_diameter_fullres']*scalefactors_dict[ID]['tissue_hires_scalef']/2
        
        
    else:
        image_dict_to_use = images_dict_10x
        # columns of adata_obj.obs table to take spot coordinates from 
        col_x = 'x_high_10X'
        col_y = 'y_high_10X'
        
        # radius of dots to plot, in pixels
        dot_radius = scalefactors_dict[ID]['spot_diameter_fullres']*scalefactors_dict[ID]['tissue_hires_scalef']/2
    
    
    if spotfilt is not None:
        print('filtering spots by area')
        if (spotfilt['row_min'] < 0) or (spotfilt['col_min'] < 0) or (spotfilt['row_max'] > adata_obj.obs.row.max()) or (spotfilt['col_max'] > adata_obj.obs.col.max()):
            print('check spotfilt - input spot filter is outside the range')
            print('# of spot rows', adata_obj.obs.row.max())
            print('# of spot columns', adata_obj.obs.col.max())
        
        # select dots
        spots = adata_obj.obs[(adata_obj.obs.row > spotfilt['row_min'])
                         &(adata_obj.obs.row < spotfilt['row_max'])
                         &(adata_obj.obs.col > spotfilt['col_min'])
                         &(adata_obj.obs.col < spotfilt['col_max'])]
        # crop image
        crop_coord = {
            'x_min': np.floor(spots[col_x].min()),
            'x_max': np.floor(spots[col_x].max()),
            'y_min': np.floor(spots[col_y].min()),
            'y_max': np.floor(spots[col_y].max())
        }
        sel_img = image_dict_to_use[ID][:, np.arange(crop_coord['x_min'], crop_coord['x_max']).astype('int64'), :]
        sel_img = sel_img[np.arange(crop_coord['y_min'], crop_coord['y_max']).astype('int64'), :, :]
        
        # add coordinates in the cropped image
        spots['x_sel'] = spots[col_x] - crop_coord['x_min']
        spots['y_sel'] = spots[col_y] - crop_coord['y_min']
        
    else:
        spots = adata_obj.obs
        spots['x_sel'] = spots[col_x]
        spots['y_sel'] = spots[col_y]
        sel_img = image_dict_to_use[ID]
        
    if filter_factor is not None:
        print('filtering spots by some key')
        spots_to_select = list(set(filter_factor.index) & set(spots.index))
        spots = spots.loc[spots_to_select,:]
        print('selected spots table: ', spots.shape)

    
    
    if spots_display == True: 


        if metric == None:
            # if no metric is passed, just plot all spots in 1 color, white, for example
            colors = [spot_color for i in range(len(spots))]
        elif metric in list(spots.columns):
            colors = list(spots[metric])
        elif metric in list(adata_obj.var_names):
            # taking normalised and log transformed expression values from .raw.X
            colors = adata_obj.raw.X[:,list(adata_obj.raw.var_names).index(metric)].todense().tolist()
            # flaten list
            colors = [item for sublist in colors for item in sublist]
    
        fig_size = figure_size
        fig = plt.figure(figsize=fig_size)
        ах = plt.subplot(111)

        ax = plt.imshow(sel_img)
        
        x_data = spots['x_sel']
        y_data = spots['y_sel']
        
        radii = [dot_radius for i in range(len(x_data))]
        patches = []
        for x, y, r in zip(x_data, y_data, radii):
            circle = (Circle(xy=(x, y), radius = r))
            patches.append(circle)
        
        colors_circles = colors
        
        # if the metric is Cluster, take colors from the jet colormap
        if metric_type == 'discrete':
            color_map = color_map_clusters
        
            # in case of custom color map
            if color_map_clusters != plt.cm.jet:
                color_map = ListedColormap(color_map)
        
        p = PatchCollection(patches, alpha=transparency_coeff, 
                            cmap=color_map)
        p.set_array(np.array(colors_circles))
        ax2 = plt.axes()
        ax2.add_collection(p)
        
        # plot the sample - may need adjustment if the figure size is changed
        #ax = plt.text(x=100,y=100,s=ID, size=20)

        # if the metric is n_genes, other QC or a marker gene expression
        if metric_type == 'continuous':
            # Add a colorbar
            cbar = fig.colorbar(p, ax=ax2)
            #cbar = plt.clim(min(colors), max(colors))

            cbar.set_label(metric,fontsize=16)

            # set the color limits
            if col_max == None:
                clim_max = max(colors)
            if col_min == None:
                clim_min = 0
                
            cbar.set_clim(col_min, col_max)
            
        if metric_type == 'discrete':
            cmap = color_map  # define the colormap
            
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]

            # create the new map
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            # np.arange doesn't include the stop value hence the +1
            bounds = np.arange(min(spots[metric])-0.5,max(spots[metric])+0.5+1,1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # create a second axes for the colorbar
            cax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.8)
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                spacing='proportional', 
                ticks=np.linspace(min(spots[metric]),max(spots[metric]),len(np.unique(spots[metric]))), 
                                           boundaries=bounds, format='%1i')

            cb.set_label(metric,fontsize=16)

        if save_pdf_name is not None:
            plt.savefig(save_pdf_name)
        
    else:
        fig = plt.figure(figsize=figure_size)
        ах = plt.subplot(111)

        ax = plt.imshow(sel_img)