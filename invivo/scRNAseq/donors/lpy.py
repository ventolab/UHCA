import numpy as np
import pandas as pd
import scanpy as sc
import scipy, random
#import scrublet as scr
from glob import iglob
import matplotlib.pyplot as plt
import anndata
#import bbknn
import os
import sklearn
import matplotlib as mpl
import pickle 
import scipy as sci
import os, sys, imp, math, re
from matplotlib.colors import LinearSegmentedColormap
from scvi.dataset import AnnDatasetFromAnnData, GeneExpressionDataset, DownloadableAnnDataset, Dataset10X
from scvi.models import TOTALVI
from scvi.inference import TotalPosterior, TotalTrainer
from scvi import set_seed
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
import ipywidgets as wigdets
import umap
#from umap import UMAP
#from umap.umap_ import UMAP
import warnings; warnings.simplefilter('ignore')
from scipy.io import mmread
from scipy.io import mmwrite
import inspect


def RTable(frame, colA, colB, colC = None):
    counts = frame.groupby([colA,colB]).size()
    matrix = pd.DataFrame(0,index = frame[colA].value_counts().index, columns = frame[colB].value_counts().index)
    for x,y in counts.index:
        matrix.at[x,y] = counts[x,y]
    return matrix

def Rmatch(la, lb):
    return [ lb.index(x) if x in lb else None for x in la ]

def reorderCat(cat, order= None, doinspect = False):
    if isinstance(cat, pd.core.arrays.categorical.Categorical) is False:
        cat = pd.Categorical(cat)
    
    if order is None:
        return pd.DataFrame(cat.categories)
    output = pd.Categorical(cat, cat.categories[order], ordered=True)
    print(pd.DataFrame(output.categories))
    return output

def applyFactorRename(input, dico, doinspect = False):
    if doinspect is True: print("\033[35;46;1mRename classes in list\033[0m\033[34m"); print(inspect.getsource(applyFactorRename));
    import numpy as np
    tmp = np.array(input, dtype=object)
    out = tmp.copy()
    for k,x in dico.items():
        if isinstance(k, tuple):
            for l in k:
                out[tmp == l] = x
        else:
            out[tmp == k] = x
    leftover_categories = set(out)
    leftover_categories = leftover_categories.difference(set(dico.values()))
    categories = list(dico.values())
    categories = categories + list(leftover_categories)
    return(pd.Categorical(out, categories, ordered=True))

def rgb(r,g,b):
    return "#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255))
def getHCYcolor(h,c,y,gam=2.2,opt_hue=True, alphablend = None):
    if isinstance(h, list):
        fout = [getHCYcolor(h[i],c[i],y[i],gam,opt_hue=opt_hue,alphablend=alphablend) for i in range(len(h))]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(len(fout)):
            bp = ax.bar(i, 1, color= fout[i])
        ax.set_ylim(0, 1)
        plt.show()
        return fout
    y = float(y);    c = float(c);    h = float(h);    y = math.pow(y, gam)
    if opt_hue is True: h = h - math.sin(h *18.8495559215) / 24
    res = (h % 1) * 6
    bas = int(res)
    res = res - bas
    if bas == 0: fout = [1.0, 1.0 - c * (1-res), 1.0 - c]
    elif bas == 1: fout = [1.0 - c * res, 1.0, 1.0 - c]
    elif bas == 2: fout =[1.0 - c, 1.0, 1.0 - c* (1-res) ]
    elif bas == 3: fout = [1.0-c,1.0 - c*res, 1.0 ]
    elif bas == 4: fout = [1.0 - c * (1-res), 1.0 - c,1.0]
    else: fout = [1.0, 1.0 - c, 1.0-c*res]
    brscale = fout[0] * 0.3 + fout[1] * 0.59 + fout[2] * 0.11
    if brscale < y:
        # need to project to be within, fit parabola f[0] = brscale, f[1] = 1 f'[x] =:
        if ((fout[0] < 1)&(fout[0] >0.00001)):
            tmp = fout[0]; fout[0] = math.pow(fout[0], 1.0/gam)
            deriv = 0.3*(1.0/fout[0]-1)*tmp;
        else: deriv =0
        if ((fout[1] < 1)&(fout[1] >0.00001)): tmp = fout[1]; fout[1] = math.pow(fout[1], 1.0/gam) ; deriv += 0.59*(1.0/fout[1]-1)*tmp;
        if ((fout[2] < 1)&(fout[2] >0.00001)): tmp = fout[2]; fout[2] = math.pow(fout[2], 1.0/gam) ; deriv += 0.11*(1.0/fout[2]-1)*tmp;
        deriv = deriv * gam
        discr = deriv * deriv - 4.0 * (brscale -y) * (1.0 -brscale - deriv)
        brscale = (-deriv + math.sqrt(discr)) / (2.0* (1.0 -brscale - deriv))
        
        fout[0] += (1 -fout[0]) * brscale
        fout[1] += (1 -fout[1]) * brscale
        fout[2] += (1 -fout[2]) * brscale
        #brscale = (fout[1]^g) * 0.3 + (fout[2]^g) * 0.59 + (fout[3]^g) * 0.11
        #if (is.null(alphablend))
        return rgb(fout[0], fout[1],fout[2])
#        else return(rgb(alphablend[4] * alphablend[1] + (1 - alphablend[4]) * (fout[1]),alphablend[4] * alphablend[2] + (1 - alphablend[4]) *  (fout[2]),alphablend[4] * alphablend[3] + (1 - alphablend[4]) * (fout[3])))
        #return rgb(0.5,0.5,0.5)
    else:
        brscale =  y/brscale
        gam = 1 / gam
#        if (!is.null(alphablend)) return(rgb(alphablend[4] * alphablend[1] + (1 - alphablend[4]) * ((fout[1]* brscale) ^ g), alphablend[4] * alphablend[2] + (1 - alphablend[4]) * ((fout[2]* brscale) ^ g),alphablend[4] * alphablend[3] + (1 - alphablend[4]) * ((fout[3]* brscale) ^ g)))
        return(rgb(math.pow(fout[0]* brscale, gam), math.pow(fout[1]* brscale, gam),math.pow(fout[2]* brscale, gam)))
def mydoublerainbow(sizelist, selectedHues=None, selectedSats=None, huerange = [0, 360], permutation=None,do_optimize_hues=True):
    nbring = len(sizelist)
    if permutation is not None:
        if (len(permutation) != len(sizelist)): permutation = range(len(sizelist))
    else: permutation = range(len(sizelist))
    if selectedHues is None:
        n = 0; nd = 0
        for j in range(nbring): n = n + 1 + (sizelist[j]/3)
        selectedHues = np.zeros(nbring)
        for j in range(nbring):
            selectedHues[permutation[j]] = (nd  * (huerange[1]- huerange[0]) / n) + huerange[0]
            nd += 1 + (sizelist[permutation[j]]/3)
            #midL = 20 * (0.25*math.sin(selectedHues * 0.01745329251994329576923690768489) - math.cos(selectedHues * 0.05235987755982988730771072305466))
    if (do_optimize_hues): selectedHues = [x - 15 * math.sin(x *0.05235987755982988730771072305466) for x in selectedHues]
    sat =1
    fout = []
    for j in range(nbring):
        phase = math.pi / sizelist[j];
        term = math.pi / sizelist[j];
        if selectedSats is not None: sat = selectedSats[j]
        for i in range(sizelist[j]):
            midL = pow(0.525 - (0.45 - 1 / (1 + sizelist[j])) * math.cos( phase + term*i) , 1.5)
            fout.append(getHCYcolor(selectedHues[j]/360, sat , midL,opt_hue=False ))
    return(fout)


def importAnndataFromFolder(prefix , obsm_names = None):
    meta = pd.read_csv(prefix + "meta.csv", index_col=0)
    feat = pd.read_csv(prefix + "feature.csv", index_col=0, header=None)
    matrix = sci.io.mmread(prefix + "raw.mtx")

    adata = anndata.AnnData(matrix.transpose(), obs= meta, var= feat)
#    if obsm_names is not None:
#        for x in obsm_names:
    return adata;

def exportAnndataToFolder(adata, prefix, filter=["data", "meta","repr", "names"]):
    if "meta" in filter:
        adata.obs.to_csv(prefix + "meta.csv")
    if "names" in filter:
        ffile = open(prefix + "feature.csv", "w")
        ffile.writelines([x + '\n' for x in adata.var.index])
        ffile.close()
    if "data" in filter:
        sci.io.mmwrite(prefix+"raw.mtx",adata.X.transpose())


def runStandardScanpy(adata, filter_obsattr = None):
    if filter.obsattr is None:
        flt = [True for i in adata.obs_names()]
    else:
        flt = [x is False for x in adata.obs[filter_obsattr].tolist()]


# concatenate rows of np2Darrays with possible colname mismatches
def cbind_ndarrays(nparr_list, colnames_list):
    outcoln = set({})
    for i in range(len(nparr_list)):
        outcoln.add(colnames_list[i])
    outcoln = list(outcoln)
    return {data: nbarr_list, colnames: outcoln}

#
#def anndata_concat(anndata_list, prefix_list, obs_annot_name = "batch"):
#    newkeys = []
#    colnames = []
#    for i in range(0, len(anndata_list)):
#        newkeys.extend([prefix[i] + "_" + s for s anndata_list[i].obs_names])
#        colnames.append(anndata_list.var_names())
#    return newkeys

def createAnnData(folderlist, prefix, souporcell_folderlist = None, souporcell_genodico = None, autoinclude=["percent_mito", "log2p1_count", "n_genes"], min_UMI_prefilter = None, min_cell_per_gene_allowed=3, min_gene_per_cell_allowed=500, sample_obskey = "sample_names",doqcplots=False, doinspect=False, mitogeneprefix="MT-", do_log2_normalize=True):
    if doinspect is True: print("\033[35;46;1mCreate AnnData Object\033[0m\033[34m"); print(inspect.getsource(createAnnData));print("\033[31;43;1mExecution:\033[0m")
    adatas = []
    def pickname(x,y):
        if (y == "singlet"): return("_genotype_" +str(x))
        else: return("_doublet")
    def pickname2(x,y,z):
        if (y == "singlet"): return(z[int(x)])
        else: return("doublet")
    for i in range(len(folderlist)):
        print("Processing " + prefix[i])
        adatas.append(sc.read_10x_mtx(folderlist[i]))
        print(adatas[-1].X.shape)
        if min_UMI_prefilter is not None: sc.pp.filter_cells(adatas[-1], min_counts= min_UMI_prefilter, inplace=True)
        print(adatas[-1].X.shape)
        if souporcell_folderlist is not None:
            try:
                res = pd.read_csv(souporcell_folderlist[i] + "clusters.tsv",sep='\t')
                if (res.shape[0] != len(adatas[i].obs_names)): raise ValueError( prefix[i] + " data and soupx barcodes mismatches!")
                adatas[i].obs["souporcell"] = res.status.to_list()
                if (souporcell_genodico is None) or (prefix[i] not in souporcell_genodico.keys()):
                    adatas[i].obs["demultiplexed"] = [prefix[i] + pickname(x,y)   for x,y in zip( res.assignment.to_list() , res.status.to_list()) ]
                else:
                    adatas[i].obs["demultiplexed"] = [pickname2(x,y, souporcell_genodico[prefix[i]] )   for x,y in zip( res.assignment.to_list() , res.status.to_list()) ]
            except:
                print("No valid genotyping data for "+ str(len(adatas[i].obs_names)) + " cells within " + prefix[i] + "! Setting everything to singlets")
                adatas[i].obs["souporcell"] = "singlet"
                if (souporcell_genodico is None):
                    adatas[i].obs["demultiplexed"] = prefix[i] + "_genotype_0"
                else:
                    adatas[i].obs["demultiplexed"] = souporcell_genodico[prefix[i]][0]
                    
        adatas[i].obs[sample_obskey] = [prefix[i] for hehe in adatas[i].obs_names]
        adatas[i].obs_names = [prefix[i] + "_" + x for x in adatas[i].obs_names]
        if "n_count" in autoinclude:
            adatas[i].obs['n_count'] = np.sum(adatas[i].X, axis=1).A1
        if "log2p1_count" in autoinclude:
            adatas[i].obs['log2p1_count'] = np.log1p(np.sum(adatas[i].X, axis=1).A1) / math.log(2)
        if "percent_mito" in autoinclude:
            mito_genes = [name for name in adatas[i].var_names if name.startswith(mitogeneprefix)]
            adatas[i].obs['percent_mito'] = np.sum(adatas[i][:, mito_genes].X, axis=1).A1 / np.sum(adatas[i].X, axis=1).A1
        if "n_genes" in autoinclude:
            adatas[i].obs['n_genes'] = np.sum(adatas[i].X != 0, axis=1).A1
        adatas[i].obs[sample_obskey].value_counts()
    adata = mergeAnnData(adatas,index_unique = None)
    if min_gene_per_cell_allowed is not None: sc.pp.filter_cells(adata, min_genes=min_gene_per_cell_allowed)
    if min_cell_per_gene_allowed is not None: sc.pp.filter_genes(adata, min_cells=min_cell_per_gene_allowed)
    
    if doqcplots is True:
        if "log2p1_count" in autoinclude:
            sc.pl.violin(adata, ['log2p1_count'], jitter=0.4, groupby=sample_obskey, rotation=90)
        if "n_genes" in autoinclude:
            sc.pl.violin(adata, ['n_genes'], jitter=0.4, groupby=sample_obskey, rotation=90)
        if "percent_mito" in autoinclude:
            sc.pl.violin(adata, ['percent_mito'], jitter=0.4, groupby=sample_obskey, rotation=90)
    if do_log2_normalize:
        adata.raw = adata.copy()
        adata.X = np.log1p(adata.X) / math.log(2)
    if "log2p1_count" in autoinclude:
        adata.obs["log2p1_count"] = pd.to_numeric(adata.obs["log2p1_count"], downcast='float')
    if "percent_mito" in autoinclude:
        adata.obs["percent_mito"] = pd.to_numeric(adata.obs["percent_mito"], downcast='float')
    return adata;

def addMetadata(adata, metadata, obs_key, meta_key, doinspect=False):
    if doinspect is True: print("\033[35;46;1mAdd metadata from sample table to anndata\033[0m\033[34m"); print(inspect.getsource(addMetadata));print("\033[31;43;1mExecution:\033[0m")
    aslist = metadata[meta_key].tolist()
    rowmap = {i : aslist.index(i)  for i in aslist}
    for val in metadata.columns:
        if val != meta_key:
            aslist = metadata[val].tolist()
            adata.obs[val] = [aslist[rowmap[i]] for i in adata.obs[obs_key] ]
    return adata;

# Merge adata objection, *including* obsm field...
def mergeAnnData(adatas, join='inner', batch_key=None, batch_categories=None, uns_merge=None, index_unique='-', fill_value=None):
    if (len(adatas) == 1): return(adatas[0])
    if batch_key is None:
        output = adatas[0].concatenate(adatas[1:len(adatas)], join=join, uns_merge=uns_merge, index_unique= index_unique, fill_value=fill_value)
    else:
        output = adatas[0].concatenate(adatas[1:len(adatas)], join=join, batch_key= batch_key, uns_merge=uns_merge, index_unique= index_unique, fill_value=fill_value)
    return output



# Split adata object using an obs attribute, uns fields are also split if the rownumber match, give an explicit list of field to overwrite this behavior
# anndata adata; input to be partitionned
# string obs_attribute_name; a key in anndata.obs
# List/Array/Set valueset; list of value for which a individual subsets are created, (default, lists all) 
def splitAnnData(adata, obs_attribute_name, valueset = None, entryfilter = None, getnames=False, forcetodense=False, use_raw_slot_instead=False, min_cell_threshold = 0, merge_toosmall_samples_instead=False, doinspect=False):
    if doinspect is True: print("\033[35;46;1mSplit AnnData into a list\033[0m\033[34m"); print(inspect.getsource(splitAnnData));print("\033[31;43;1mExecution:\033[0m")
    if entryfilter is None:
        entryfilter = [True for sillypython in adata.obs_names]
    elif (isinstance(entryfilter, tuple)): entryfilter = list(entryfilter)
    
    if obs_attribute_name is None:
        postflts = [ entryfilter ]
        postvalueset = ["everything"] 
    else:
        if valueset is None:
            valueset = list(set(adata.obs[obs_attribute_name][entryfilter]))
        flts = [adata.obs[obs_attribute_name] == s  for s in valueset]
        postflts = []
        postvalueset = []
        for i in range(len(flts)):
            if (sum(flts[i] & entryfilter) >= min_cell_threshold):
                postflts.append(flts[i])
                postvalueset.append(valueset[i])
    which = lambda lst:list(np.where(lst)[0])
    wlts = [ which(f & entryfilter) for f in postflts]
    
    if use_raw_slot_instead is False:
        if isinstance(adata.X , scipy.sparse.coo.coo_matrix):
            ref = adata.X.tocsr()
            if forcetodense is True:
                output =  [ anndata.AnnData(ref[wlts[f],].todense(), obs = adata.obs.iloc[wlts[f]], var = adata.var, varm = adata.varm, varp=adata.varp, uns = adata.uns) for f in range(len(postflts)) ]
            else:
                output =  [ anndata.AnnData(ref[wlts[f],], obs = adata.obs.iloc[wlts[f]], var = adata.var, varm = adata.varm, varp=adata.varp, uns = adata.uns) for f in range(len(postflts)) ]
        else:
            output =  [ anndata.AnnData(adata.X[wlts[f],], obs = adata.obs.iloc[wlts[f]], var = adata.var, varm = adata.varm, varp=adata.varp, uns = adata.uns) for f in range(len(postflts)) ]
    else:
        if isinstance(adata.raw.X , scipy.sparse.coo.coo_matrix):
            ref = adata.raw.X.tocsr()
            if forcetodense is True:
                output =  [ anndata.AnnData(ref[wlts[f],].todense(), obs = adata.obs.iloc[wlts[f]], var = adata.var, varm = adata.varm, varp=adata.varp, uns = adata.uns) for f in range(len(postflts)) ]
            else:
                output =  [ anndata.AnnData(ref[wlts[f],], obs = adata.obs.iloc[wlts[f]], var = adata.var, varm = adata.varm, varp=adata.varp, uns = adata.uns) for f in range(len(postflts)) ]
        else:
            output =  [ anndata.AnnData(adata.raw.X[wlts[f],], obs = adata.obs.iloc[wlts[f]], var = adata.var, varm = adata.varm, varp=adata.varp, uns = adata.uns) for f in range(len(postflts)) ]        

    for k in  adata.obsm:
        for o in range(len(output)):
            output[o].obsm[k] = adata.obsm[k][np.ix_(wlts[o], range(adata.obsm[k].shape[1]))]
    
    if (getnames is False): return output
    danames= list(output[0].obs_names)
    for i in range(1, len(output)):
        danames = danames + list(output[i].obs_names)
    return{"datalist" : output, "ordered" : danames, "subsets" : list(postvalueset)} 

def bh(pvalues):
    """
    Computes the Benjamini-Hochberg FDR correction.
    Input:
    * pvals - vector of p-values to correct
    """
    pvalues = np.array(pvalues)
    n = int(pvalues.shape[0])
    new_pvalues = np.empty(n)
    values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]
    values.sort()
    values.reverse()
    new_values = []
    for i, vals in enumerate(values):
        rank = n - i
        pvalue, index = vals
        new_values.append((n/rank) * pvalue)
    for i in range(0, int(n)-1):
        if new_values[i] < new_values[i+1]:
            new_values[i+1] = new_values[i]
    for i, vals in enumerate(values):
        pvalue, index = vals
        new_pvalues[index] = new_values[i]
    return new_pvalues

def bonf(pvalues):
    """
    Computes the Bonferroni FDR correction.

    Input:

    * pvals - vector of p-values to correct
    """
    new_pvalues = np.array(pvalues) * len(pvalues)
    new_pvalues[new_pvalues>1] = 1
    return new_pvalues

def scrub(adata, batch_obsattrib, bonf_threshold = 0.01, add_qc_metrics=False,mito_prefix= "MT-", obskey_cellfilter = "filtered_cells", add_cell_filter={"max_percent_mito": 0.15, "scrublet_local_pred": False}, doinspect=False):
    if doinspect is True: print("\033[35;46;1mDetect Doublets and defining cells to filter\033[0m\033[34m"); print(inspect.getsource(scrub));print("\033[31;43;1mExecution:\033[0m")
    
    import scrublet as scr
    import scanpy as sc
    print("spliting data using attribute " + batch_obsattrib)
    adatas = splitAnnData(adata, batch_obsattrib)

    if (add_qc_metrics):
        mito_genes = [name for name in adata.var_names if name.startswith(mito_prefix)]
        adata.obs['log2p1_RNA_count'] = np.log1p(adata.X.sum(axis=1).A1) / math.log(2)
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1

    dalist = list(adata.obs_names)
    tmap = {}
    for i in range(len(adata.obs_names)):
        tmap.update( {adata.obs_names[i] : i})
    adata.obs["scrublet_pred"] = False
    adata.obs["scrublet_local_pred"] = False
    adata.obs["scrublet_score"] = 0.0
    adata.obs["scrublet_cluster_score"] = 0.0

    # Luz's double clustering approach
    for i in range(len(adatas)):

        print("processing " + str(i) + "/" + str(len(adatas)))
        curscr = scr.Scrublet(adatas[i].X)
        doublet_scores, predicted_doublets = curscr.scrub_doublets()
        adatas[i].obs['scrublet_score'] = doublet_scores
        adatas[i].obs['scrublet_pred'] = predicted_doublets
        #overcluster prep. run turbo basic scanpy pipeline
        sc.pp.normalize_per_cell(adatas[i], counts_per_cell_after=1e4)
        sc.pp.log1p(adatas[i])
        sc.pp.highly_variable_genes(adatas[i], min_mean=0.0125, max_mean=3, min_disp=0.5)
        adatas[i] = adatas[i][:, adatas[i].var['highly_variable']]
        sc.pp.scale(adatas[i], max_value=10)
        sc.tl.pca(adatas[i], svd_solver='arpack')
        sc.pp.neighbors(adatas[i])
        #overclustering proper - do basic clustering first, then cluster each cluster
        sc.tl.leiden(adatas[i])
        adatas[i].obs['leiden'] = [str(i) for i in adatas[i].obs['leiden']]
        for clus in np.unique(adatas[i].obs['leiden']):
            adata_sub = adatas[i][adatas[i].obs['leiden']==clus].copy()
            sc.tl.leiden(adata_sub)
            adata_sub.obs['leiden'] = [clus+','+i for i in adata_sub.obs['leiden']]
            adatas[i].obs.loc[adata_sub.obs_names,'leiden'] = adata_sub.obs['leiden']

        #compute the cluster scores - the median of Scrublet scores per overclustered cluster
        for clus in np.unique(adatas[i].obs['leiden']):
            adatas[i].obs.loc[adatas[i].obs['leiden']==clus, 'scrublet_cluster_score'] = \
                np.median(adatas[i].obs.loc[adatas[i].obs['leiden']==clus, 'scrublet_score'])
        #now compute doublet p-values. figure out the median and mad (from above-median values) for the distribution
        med = np.median(adatas[i].obs['scrublet_cluster_score'])
        mask = adatas[i].obs['scrublet_cluster_score']>med
        mad = np.median(adatas[i].obs['scrublet_cluster_score'][mask]-med)
        #let's do a one-sided test. the Bertie write-up does not address this but it makes sense
        zscores = (adatas[i].obs['scrublet_cluster_score'].values - med) / (1.4826 * mad)
        #adatas[i].obs['zscore'] = zscores
        pvals = 1-scipy.stats.norm.cdf(zscores)
        #adatas[i].obs['bh_pval'] = bh(pvals)
        pvals = bonf(pvals)
        adatas[i].obs['scrublet_local_pred'] = pvals < bonf_threshold
        map = [tmap[s] for s in adatas[i].obs_names]
        print("annoying values")
        for j in range(len(adatas[i].obs_names)):
            adata.obs["scrublet_pred" ][map[j]] = adatas[i].obs["scrublet_pred"][j]
            adata.obs["scrublet_local_pred"][map[j]] = adatas[i].obs["scrublet_local_pred"][j]
            adata.obs["scrublet_cluster_score"][map[j]] =  adatas[i].obs["scrublet_cluster_score"][j]
            adata.obs["scrublet_score"][map[j]] =  pvals[j]

    adata.obs["scrublet_cluster_score"] = pd.to_numeric(adata.obs["scrublet_cluster_score"], downcast='float')
    adata.obs["scrublet_score"] = pd.to_numeric(adata.obs["scrublet_score"], downcast='float')

    if add_cell_filter is not None:
        curflt = np.zeros(adata.obs.shape[0], dtype=bool) 
        for k,v in add_cell_filter.items():
            if k == "max_percent_mito":
                print(str(sum(adata.obs['percent_mito'] > v)) + " cells filtered by mito threshold")
                curflt |= adata.obs['percent_mito'] > v
            else:
                print(str(sum(adata.obs[k] != v)) + " cells filtered by " + k + " == " + str(v) + " criterion")
                curflt |= (adata.obs[k] != v)
        adata.obs[obskey_cellfilter] = curflt
    return adata


#string obskey; instead of using adata.X for clustering, use "PCs" presumably stored in adata.obsm[obskey]
def scanpy_subclustering(adata, partition, clusterkey, cell_filter = None, resolution = 1, obskey = None):
    print("Spliting data")
    adatas = splitAnnData(adata, partition, entryfilter=cell_filter, getnames = True)
    scanpy_arr = []
    dalist = list(adata.obs_names)
    tmap = {}
    iswithin = np.zeros(len(adata.obs_names), dtype=bool)
    for i in range(len(adatas["datalist"])):
        for j in range(len(adatas["datalist"][i].obs_names)):
            tmap.update( {adatas["datalist"][i].obs_names[j] : j + len(scanpy_arr)})
        print("(" + str(i) + "/" + str(len(adatas["datalist"]) ) + ") Processing " +adatas["subsets"][i] + ": " + str(len(adatas["datalist"][i].obs_names)) +" cells", end = '')
        iswithin |= adata.obs[partition] == adatas["subsets"][i] 
        if obskey is None:
            sc.pp.normalize_per_cell(adatas["datalist"][i], counts_per_cell_after=1e4)
            sc.pp.log1p(adatas["datalist"][i])
            sc.pp.highly_variable_genes(adatas["datalist"][i], min_mean=0.0125, max_mean=3, min_disp=0.5)
            adatas["datalist"][i] = adatas["datalist"][i][:, adatas["datalist"][i].var['highly_variable']]
            sc.pp.scale(adatas["datalist"][i], max_value=10)
            sc.tl.pca(adatas["datalist"][i], svd_solver='arpack')
            sc.pp.neighbors(adatas["datalist"][i])
        else:
            sc.pp.neighbors(adatas["datalist"][i], use_rep =obskey)
        #overclustering proper - do basic clustering first, then cluster each cluster
        sc.tl.leiden(adatas["datalist"][i],resolution=resolution)
        scanpy_arr += [adatas["subsets"][i] + "_cl" + str(j) for j in adatas["datalist"][i].obs['leiden']]
        print(" into " + str(len(set(adatas["datalist"][i].obs['leiden'])))+ " clusters")
    scanpy_clusters = np.zeros(adata.obs.shape[0] , dtype=object)
    scanpy_clusters[:] = "filtered"
    iswithin &= cell_filter
    scanpy_arr = np.array(scanpy_arr)
    scanpy_clusters[iswithin] = scanpy_arr[[tmap[s] for s in adata.obs_names[iswithin]]]
    adata.obs[clusterkey] = scanpy_clusters
    return(adata)

def addCycleCycleAnnotation(adata, s_genes = None, g2m_genes = None, geneprefix = "", use_raw_data= True, doinspect=False):
    if doinspect is True: print("\033[35;46;1mAdd Cell Cycle annotation to anndata object\033[0m\033[34m"); print(inspect.getsource(addCycleCycleAnnotation));print("\033[31;43;1mExecution:\033[0m")

    # uses Seurat Cell Cycles default genes by default
    if s_genes is None: # "MLF1IP"
        s_genes = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
    if g2m_genes is None: #use default list
        g2m_genes =["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]
    s_genes =  [geneprefix  + x for x in s_genes]
    g2m_genes =  [geneprefix  + x for x in g2m_genes]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, use_raw = use_raw_data)
    return(adata)

def scvi_prepare(anndatapath, field, cellfilter = None, nbgenes = 5000, genes_to_filter= None, use_ccfilter_prefix=None, citeseqkey = "protein_expression", use_raw_slot_instead =None, min_cell_threshold= 0, doinspect=False):
    if doinspect is True: print("\033[35;46;1mPrepare Data for Scvi/TotalVi\033[0m\033[34m"); print(inspect.getsource(scvi_prepare));print("\033[31;43;1mExecution:\033[0m")
    if use_ccfilter_prefix is not None :
        genes_to_filter = [use_ccfilter_prefix + x for x in ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA","MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]]
    if (isinstance(anndatapath, str)): cite = anndata.read_h5ad(anndatapath)
    else: cite = anndatapath
    if use_raw_slot_instead is None: # default behavior, check for the existance of the raw slot
        use_raw_slot_instead = cite.raw is not None
    
    split = splitAnnData(cite, field, entryfilter= cellfilter, getnames = True, use_raw_slot_instead= use_raw_slot_instead,min_cell_threshold=min_cell_threshold)
    if genes_to_filter is not None:
        for i in range(len(split["datalist"])):
            split["datalist"][i] = split["datalist"][i][:, [g not in genes_to_filter for g in split["datalist"][i].var_names]]
    
    

    dataset = GeneExpressionDataset()
    if citeseqkey in cite.obsm.keys():
        if citeseqkey is not "protein_expression":
            for i in range(len(split["datalist"])):
                split["datalist"][i].obsm["protein_expression"] = split["datalist"][i].obsm[citeseqkey]
        dataset.populate_from_datasets([AnnDatasetFromAnnData(ad=s,cell_measurements_col_mappings={"protein_expression":"protein_names"})   for s in split["datalist"] ])
    else:
        dataset.populate_from_datasets([AnnDatasetFromAnnData(ad=s)   for s in split["datalist"] ])

    if nbgenes != 0: # nbgenes == 0 is a flag for the default behavior, which is to include all the genes
        dataset.subsample_genes(nbgenes)
    return{"dataset":dataset , "names" : split["ordered"]}


def runSCVI(dataset, nbstep = 500, n_latent = 64, doinspect= False):
    if doinspect is True: print("\033[35;46;1mRun scvi\033[0m\033[34m"); print(inspect.getsource(runSCVI));print("\033[31;43;1mExecution:\033[0m")
    vae = VAE(dataset.nb_genes, n_batch= dataset.n_batches, n_labels= dataset.n_labels, n_latent = n_latent)
    trainer = UnsupervisedTrainer(vae, dataset, train_size=0.9, frequency=5, use_cuda=True)
    trainer.train(n_epochs=nbstep)
    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    return(full.sequential().get_latent()[0])

def runTotalVi(dataset, nbstep= 500, n_latent=64, doinspect=False):
    if doinspect is True: print("\033[35;46;1mRun totalvi\033[0m\033[34m"); print(inspect.getsource(runTotalVi));print("\033[31;43;1mExecution:\033[0m")
    totalvae = TOTALVI(dataset.nb_genes, len(dataset.protein_names), n_batch=dataset.n_batches, n_latent=n_latent)
    lr = 4e-3
    test_mode = False
    # totalVI is trained on 90% of the data
    # Early stopping does not comply with our automatic notebook testing so we disable it when testing
    trainer = TotalTrainer(totalvae, dataset, train_size=0.90, test_size=0.10, use_cuda=True, frequency=1, batch_size=256, early_stopping_kwargs="auto" if not test_mode else None)
    trainer.train(lr=lr, n_epochs=nbstep)
    full = trainer.create_posterior(type_class=TotalPosterior)
    print(" suggested:  latent = full.sequential().get_latent()")
    return(full)

def insertAnnotation(adata_trg, adata_src, annotation, annotation_trg = None):
    if annotation_trg is None:
        annotation_trg = annotation
    dalist = list(adata_trg.obs_names)
    tmap = {}
    for i in range(len(adata_trg.obs_names)):
        tmap.update( {adata_trg.obs_names[i] : i})
    map = [tmap[s] for s in adata_src.obs_names]
    adata_trg.obs[annotation_trg] = ""
    for i in range(len(map)):
        adata_trg.obs[annotation_trg][map[i]] = adata_src.obs[annotation][i]
    return(adata_trg)

# inserts latent variable back into adata, and include umap and clusters (both disabled with umap_key= None)
def insertLatent(adata, latent , latent_key= "latent", umap_key= "X_umap", tsne_key = "X_tsne", leiden_key = "leiden", rename_cluster_key= None,cellfilter = None, cellnames =None, leiden_resolution=1.0,doinspect=False):
    if doinspect is True: print("\033[35;46;1mCompute Clusters and Reduces representations\033[0m\033[34m"); print(inspect.getsource(insertLatent));print("\033[31;43;1mExecution:\033[0m")

    if cellnames is None:
        #order of full must match
        assert latent.shape[0] == len(adata.obs_names),  "cell names need for be provided if size of latent mismatches adata"
        map = range(len(adata.obs_names))
        if latent_key is not None:
            adata.obsm[latent_key] = latent
    else:
        print("defining permutation")
        dalist = list(adata.obs_names)
        tmap = {}
        for i in range(len(adata.obs_names)):
            tmap.update( {adata.obs_names[i] : i})
# for i in range(len(adata.obs_names))}
        map = [tmap[s] for s in cellnames]
        if latent_key is not None:
            print("Inserting Latent coords")
            adata.obsm[latent_key] = np.zeros( (len(adata.obs_names), latent.shape[1]) )
            for i in range(len(map)):
                adata.obsm[latent_key][map[i],:] = latent[i,:]
    
    if umap_key is not None:
        import umap.umap_ as umap
        print("computing UMAP")
        tumap = umap.UMAP(spread=2).fit_transform(latent)
        adata.obsm[umap_key] = np.zeros( ( len(adata.obs_names), 2 ) )

        print("Inserting Umap coords")
        for i in range(len(map)):
            adata.obsm[umap_key][map[i],:] = tumap[i,:]

    if tsne_key is not None:
        import scanpy as sc
        print("computing Tsne")
        adata_latent = sc.AnnData(latent)

        sc.tl.tsne(adata_latent, use_rep='X', n_jobs=8)
        adata.obsm[tsne_key] = np.zeros( ( len(adata.obs_names), 2 ) )

        print("Inserting Tsne coords")
        for i in range(len(map)):
            adata.obsm[tsne_key][map[i],:] = adata_latent.obsm["X_tsne"][i,:]

    if leiden_key is not None:
        import scanpy as sc
        adata_latent = sc.AnnData(latent)
        print("Finding clusters")
        sc.pp.neighbors(adata_latent, use_rep='X', n_neighbors=30, metric='minkowski')
        sc.tl.leiden(adata_latent, resolution=leiden_resolution)
        adata.obs[leiden_key] = "filtered"
        #if rename_cluster_key is not None:
        #    print("Renaming to do...")
        #    for i
        #else
        #    ctnames = range(200)
        
        print("Inserting Cluster Id")
        for i in range(len(map)):
            adata.obs[leiden_key][map[i]] = adata_latent.obs["leiden"][i]
    return adata

def makeCellxgeneFriendly(adata):
    import math
    if adata.raw is None:
        adata.raw = anndata.AnnData(adata.X)
        adata.X = np.log1p(adata.X) / math.log(2)
    if "nCount_RNA" in adata.obs.keys():
        adata.obs["log2p1_RNAcount"] = np.log1p(adata.obs["nCount_RNA"]) / math.log(2)
    


def findCoexpressFilter(adata, prior_genes, prior_gene_count_thresholds = None, positive_cell_annotation = None, do_just_count=False, threshold = None):
    import rbcde
    if prior_gene_count_thresholds is None: prior_gene_count_thresholds = [1 for x in prior_genes]
    if positive_cell_annotation is None: conv = "thistemporaryannotationnameistoowierdtobealreadyincludedintheinputobject"
    else: conv = positive_cell_annotation
    adata.obs[conv] = False
    tmp = adata.var_names.to_list()
    for i in range(len(prior_genes)):
        res = (adata.X[:,tmp.index(prior_genes[i])] >= prior_gene_count_thresholds[i]).tolist() 
        print(str(sum(res)) + " cells passed the threshold for ", prior_genes[i])
        adata.obs[conv] |= res
    print("Defined a cluster with " +str(sum(adata.obs[conv]))+ " cells positive for the prior genes")
    if do_just_count is True: return(adata)
    adata.obs[conv] = [str(x) for x in adata.obs[conv] ]
    rbcde.RBC(adata, use_raw=False, clus_key=conv)
    if positive_cell_annotation is None: del adata.obs[conv]
    print("R for prior genes are:")
    Rprior = adata.var.loc[prior_genes]["RBC_True"]
    print(Rprior)
    if threshold is None:
        threshold = min(Rprior)
    rcoefs = adata.var["RBC_True"] > threshold
    genelist = adata.var_names[rcoefs]
    rcoefs = adata.var["RBC_True"][rcoefs]
    del adata.var["RBC_False"]
    del adata.var["RBC_True"]
    return {"genes": genelist, "Rcoef": rcoefs} 

def findCoexpressFilterVersion2(adata, prior_genes, prior_fraction = 0.01, positive_cell_annotation = None, threshold= None):
    import rbcde
    import numpy as np
    import math
    if positive_cell_annotation is None: conv = "thistemporaryannotationnameistoowierdtobealreadyincludedintheinputobject"
    else: conv = positive_cell_annotation
    adata.obs[conv] = False
    tmp = adata.var_names.to_list()
    paxe = np.ones(len(prior_genes))
    cprior = adata.X[:, [tmp.index(i) for i in prior_genes]];
    cprior /= np.sum(adata.X, axis = 1)[:,np.newaxis]
    nbcells = int(prior_fraction * cprior.shape[0])
    
    for i in range(10):
        print(paxe)
        proj = np.matmul(cprior, paxe).tolist()
        whichcells = sorted(range(len(proj)), reverse=True, key=lambda k: proj[k])[0:nbcells]
        paxe = np.sum(cprior[whichcells,:],axis=0) / nbcells
    adata.obs[conv] = [i in whichcells for i in range(cprior.shape[0])]
    adata.obs[conv] = [str(x) for x in adata.obs[conv] ]
    rbcde.RBC(adata, use_raw=False, clus_key=conv)
    if positive_cell_annotation is None: del adata.obs[conv]
    print("R for prior genes are:")
    Rprior = adata.var.loc[prior_genes]["RBC_True"]
    print(Rprior)
    if threshold is None:
        threshold = min(Rprior)
    rcoefs = adata.var["RBC_True"] > threshold
    genelist = adata.var_names[rcoefs]
    rcoefs = adata.var["RBC_True"][rcoefs]
    del adata.var["RBC_False"]
    del adata.var["RBC_True"]
    return {"genes": genelist, "Rcoef": rcoefs}

def binarizedDE(adata, obskey_cluster="cluster"):
    adata_counts = anndata.AnnData(X=np.expm1(adata.raw.X), var=adata.raw.var, obs=adata.obs, asview=False)
    # Binarize counts matrix (1 if expressed, 0 otherwise)
    import episcanpy as epi
    adata_bin = epi.pp.binarize(adata_counts, copy=True)
    # Compute TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(adata_bin.X.T)
    adata_bin.X = tfidf.T
    #Compute Mann-Withney test with scanpy
    sc.tl.rank_genes_groups(adata_bin, groupby=obskey_cluster, n_genes=50000, method='wilcoxon', use_raw=False) #, method='logreg'
    #Plot TF-IDF values
    sc.pl.rank_genes_groups_dotplot(adata_bin, n_genes=20, dendrogram=False)
    # Plot with gene expression normalized values
    adata_count = anndata.AnnData(X=np.expm1(adata.raw.X), var=adata.raw.var, obs=adata.obs, asview=False)
    sc.pp.normalize_per_cell(adata_count, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_count)
    sc.pp.scale(adata_count, max_value=10)
    adata_count.uns['rank_genes_groups'] = adata_bin.uns['rank_genes_groups']
    sc.pl.rank_genes_groups_dotplot(adata_count, n_genes=15, dendrogram=False, save='_tfidf_scanpy.pdf')


def makeBernoulliMatrix(adata, obsname):
    import numpy as np
    catlist = adata.obs["donor_deconv"].cat.categories.to_list()
    fout = np.zeros([adata.X.shape[0], len(catlist)], dtype=bool)
    for i in range(len(catlist)):
        fout[:,i] = adata.obs["donor_deconv"] == catlist[i]
    return fout

def makeMofaMatrices(adata, obs_batchname, variable_genes,cellfilter= None, use_old = False, citeobsm =None):
    import scanpy as sc
    # find variable genes:
    print("LogXform")
    if isinstance(variable_genes, int):
        citeseqflt = adata.var["feature_types"] != "Antibody Capture"
        tmpadata = sc.pp.log1p(adata[:,[citeseqflt[hehe] for hehe in adata.var.index]], copy=True , base=2)
        sc.pp.highly_variable_genes(tmpadata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        variable_genes = tmpadata.var['dispersions'].nlargest(variable_genes).index
        return(variable_genes)
    
    if use_old is True:
        mofa_matrices = []
        print("Dispersion computed")
        mofa_matrices.append([tmpadata[:,[hehe in variable_genes for hehe in tmpadata.var.index]].X.todense()])
        print("Done inserting Gex data")
        mofa_matrices.append(sc.pp.log1p(adata.obsm['protein_expression'], base =2))
        print("Done inserting citeseq data")
        mofa_matrices.append(makeBernoulliMatrix(adata, obs_batchname))
        print("Done inserting batch data")
        return(mofa_matrices)
    else:
        print("splitting matrices")
        if citeobsm is None:
            citeseqflt = adata.var["feature_types"] == "Antibody Capture"
        adatas = splitAnnData(adata, obs_batchname, entryfilter=cellfilter, getnames=True)
        mofa_matrices = [[None for g in range(len(adatas["datalist"]))] for m in range(2)]
        for g in range(len(adatas["datalist"])):
            print("Insert1")
            mofa_matrices[0][g] = sc.pp.log1p(adatas["datalist"][g].X[:,[hehe in variable_genes for hehe in adatas["datalist"][g].var.index]],copy=True,base =2).todense()
            if citeobsm is None:
                mofa_matrices[1][g] = sc.pp.log1p(adatas["datalist"][g][:,[citeseqflt[hehe] for hehe in adatas["datalist"][g].var.index]],copy=True,base =2).X
            else:
                mofa_matrices[1][g] = sc.pp.log1p(adatas["datalist"][g].obsm[citeobsm])
            
            if isinstance(mofa_matrices[1][g], np.ndarray) is False:
                mofa_matrices[1][g] = mofa_matrices[1][g].todense()

        return {"matrices": mofa_matrices, "names": adatas["ordered"]}

# was 1000 iter... wont complete eh?
def runMOFA(mofamatrices, likelihoods = ["gaussian","gaussian"], outfile = "mofa_joint.hdf5", nb_factors= 64, iter=100,convergence_mode='medium'):
    #"bernoulli" 
    import mofapy2
    from mofapy2.run.entry_point import entry_point
    ent = entry_point()
    print("Set matrices")
    ent.set_data_matrix(mofamatrices, likelihoods = likelihoods)
    ent.set_model_options(factors = nb_factors)
    print("Set options")
    ent.set_train_options(iter = iter, convergence_mode = convergence_mode, startELBO = 1, freqELBO = 1, gpu_mode = True, verbose = False, seed = 1)
    print("Building!")
    ent.build()
    print("running!")
    ent.run()
    # Save the output
    print("saving!")
    ent.save(outfile)
    return ent
#.model.getExpectations()["Z"]["E"]


# enforce the categories to contain all the values in the desired order
def enforceMetadata(dataframe,dic_of_categorical_values = None):
    if dic_of_categorical_values is None:
        dic_of_categorical_values = {"Age": ["0","2","6","12","18","30","40","50","60","70","80","90", "NA"], "Sex": ["M","F", "NA"], "Race": ["White", "Black", "Asian", "Other","NA"], "Ethnicity" : ["Not Hispanic or Latino", "Hispanic or Latino", "Unknown or not documented","NA"], "BMI" : ["<18.5 (underweight)","18.5-24.9 (normal)","25.0-29.9 (overweight)","30.0-39.9 (obese)",">=40 (severely obese)","Unknown","NA"], "Highest level of respiratory support" : ["Mechanical ventilation with intubation", "Non-invasive ventilation (CPAP, BiPAP, HFNC)", "Supplemental O2", "None","NA"], "Smoking": ["Never or unknown", "Prior", "Current","NA"], "SARS-CoV-2 Ab" : ["Negative", "Positive", "Not Done","NA"], "SARS-CoV-2 PCR" : ["Positive", "Negative","NA"]}
        for c in ["Pre-existing hypertension", "Pre-existing diabetes", "Pre-existing kidney disease", "Pre-existing lung disease", "Pre-existing heart disease", "Pre-existing immunocompromised condition", "Symptomatic", "Admitted to hospital", "Vasoactive agents required during hospitalization", "28-day death"]:
            dic_of_categorical_values[c] = ["No", "Yes", "NA"]
    for k,v in dic_of_categorical_values.items():
        if k in dataframe:
            dataframe[k] = pd.Categorical(dataframe[k], categories = v, ordered=True)
    return dataframe

def populateDependentAnnotations(adata, dataframe, shared_key, yes_no_color = ["#007BDC", "#FFA000","#DDDDDD"], ico_of_colors= {}):
    aslist = dataframe[shared_key].to_list()
    coloff = [aslist.index(x)  for x in adata.obs[shared_key]]
    for col in dataframe.columns:
        if col == shared_key: continue
        if isinstance(dataframe[col], pd.core.arrays.categorical.Categorical) is False:
            # I hate python, so I need to fix it myself
            dataframe[col] = pd.Categorical(dataframe[col])
        adata.obs[col] = pd.Categorical(dataframe[col][coloff].to_list(), dataframe[col].cat.categories, ordered=True)
        if (len((adata.obs[col].cat.categories)) == 3)&(adata.obs[col].cat.categories[0] in ["No", "Negative", "no", "negative"])&(adata.obs[col].cat.categories[1] in ["Yes", "yes", "Positive", "positive"]):
            adata.uns[col + "_colors"] = yes_no_color

    # reorder columns to match
    metadatafields = adata.obs.columns[[ x not in dataframe.columns for x in adata.obs.columns]]
    adata.obs = adata.obs.reindex(metadatafields.to_list() + dataframe.columns.to_list(), axis =1)

    return adata


def myPlot(adata, value, coor, size = [6, 4], pdfsave = None):
    if value in adata.obs.keys():
        if (value + "_colors") in adata.uns.keys():
            cm = LinearSegmentedColormap.from_list('my_cm', adata.uns[value + "_colors"], N=len(adata.uns[value + "_colors"]))
        else:
            print("TODO")
            return
        fig, ax = plt.subplots(figsize=(size[0], size[1]))
        order = np.arange(adata.obs.shape[0])
        np.random.shuffle(order)
        ax.scatter(adata.obsm[coor][order,0], adata.obsm[coor][order,1],c=adata.obs[value].cat.codes[order],cmap=cm, edgecolors='none', s=5)
        plt.axis("off")
        fig.set_tight_layout(True)
        leg = []
        for i in set(adata.obs[value].cat.codes[order]):
            leg += Line2D([0],[0],marker='o', label = adata.obs[value].cat)
    else:
        print("TODO")
    #clusters = [adata.obs[viewann].cat.categories.to_list().index(c) for c in adata.obs[viewann]]
    #colors = adata.uns[viewann + "_colors"]
    #for i, k in enumerate(np.unique(clusters)):
    #    plt.scatter(prep.obsm["X_umap"][clusters == k, 0], prep.obsm["X_umap"][clusters == k, 1], label=adata.obs[viewann].cat.categories.to_list()[k],
    #            edgecolors='none', c=colors[k % 23], s=5)
    #plt.legend(borderaxespad=0, fontsize='large', markerscale=5, loc="upper right", bbox_to_anchor=(1.2, 1))
    #plt.axis('off')
    #fig.set_tight_layout(True)

    ax.legend()
    if pdfsave is not None:
        fig.savefig(pdfsave, bboc_inches='tight')

    return(ax)    



def runPalantir(adata, startcell, doinspect = False):
    if doinspect is True: print("\033[35;46;1mCompute Trajectory using Palantir\033[0m\033[34m"); print(inspect.getsource(runPalantir));print("\033[31;43;1mExecution:\033[0m")
    import palantir
    #import harmony
    if 'highly_variable' not in adata.var.keys():
        sc.pp.highly_variable_genes(adata,inplace=True)
    pca_projections, _ = palantir.utils.run_pca(adata)
    dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)
    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    pr_res = palantir.core.run_palantir(ms_data, startcell, num_waypoints=500)
    

    return(pr_res)

#def SparseMatric_toR():

#def SparseMatrix_fromR(rpy2_rs4_object):
   
# source_cells is either a list of cells within the adata object, or another adata with matching cell names
def propagateViaConnectivies(adata, obskey_dico, source_cells, target_cells ,rep):
    subadata =  adata[ target_cells ,]
    sc.pp.neighbors(subadata, n_neighbors=25, use_rep=rep)
   
    if (isinstance(source_cells, list)): source_cells =  np.array(source_cells,dtype=bool)
    if (isinstance(target_cells, list)): target_cells =  np.array(target_cells,dtype=bool)

    dif = source_cells[target_cells]
    print(len(dif))
    print(sum(dif))
    which = lambda lst:list(np.where(lst)[0])
    offset = which(target_cells)
    for k,d in obskey_dico.items():
        subadata.obs[d] = 0.0
        adata.obs[d] = 0.0
    for i in range(len(dif)):
        if dif[i] == True:
            for k,d in obskey_dico.items():
                subadata.obs[d][i] = subadata.obs[k][i] 
        else:
            sub = subadata.uns["neighbors"]["connectivities"].indices[range( subadata.uns["neighbors"]["connectivities"].indptr[i],subadata.uns["neighbors"]["connectivities"].indptr[i+1])]
            sub = sub[[ source_cells[x] for x in sub ]].tolist()
            if len(sub) > 0:
                for k,d in obskey_dico.items():
                    subadata.obs[d][i] = sum(subadata.obs[k].iloc[sub]) / len(sub)
            else: 
                for k,d in obskey_dico.items():
                    subadata.obs[d][i] = None
    print("transfer time...")
    for i in range(len(offset)):
        for k,d in obskey_dico.items():
            adata.obs[d][offset[i]] = subadata.obs[d][i]
    for k,d in obskey_dico.items():
        adata.obs[d] = pd.to_numeric(adata.obs[d], downcast='float')
    return(adata)

#sumset set of "true" entries to a selected number, for each partition if additionnaly provided
def subsample(truefalse_vector, subsamplesize, partition = None, doinspect = False):
    if doinspect is True: print("\033[35;46;1mSample a subset of a defined size\033[0m\033[34m"); print(inspect.getsource(subsample));print("\033[31;43;1mExecution:\033[0m")
    if partition is None:
        partition = ["thesame" for x in range(len(truefalse_vector))]
    valueset = list(set(partition))
    fout = np.zeros(len(truefalse_vector), dtype="bool")
    which = lambda lst:list(np.where(lst)[0])
    for i in valueset:
        subf = truefalse_vector & (partition == i)
        wlts = which(subf)
        if len(wlts) > subsamplesize:
            wlts = random.sample(wlts, subsamplesize)
        fout[wlts] = True
    return(fout)


def prePlot(adata, obskeyA, obskeyB, cellsubset = None):
    if cellsubset is None:
        adata2 = adata.copy()
    else:
        adata2 = adata[cellsubset]
    adata2.obs["dcat"] = [ x + y for x,y in zip(adata2.obs[obskeyA], adata2.obs[obskeyB]  )]
    return(adata2)

def doLogisticRegression(adata, adata_ref, annotation_src, annotation_trg = None, annotation_trg_prob = None, cell_filter_trg = None, cell_filter_src = None, regr_max_iter = 10000, use_variable_genes=True, use_variable_genes_within_subset=True,  cc_gene_filter_prefix= None, use_raw_data=True, use_max_normalization=True, do_compute_cosine_distance=True, filtered_label = "filtered", make_circular_coords=False, cosine_coor_softmax_coef= None, logistic_coor_softmax_coef = None, C_parameter = 1, genespeficityfraction=None, doinspect=False):
    """Compute Logistic regression based projection

    Parameters
    ----------
    adata : Annadata.anndata
        Object containing the cells that are projected
    adata_ref: Annadata.anndata
        Object containing the reference cells that are defining classes used for the projection
    annotation_src : string
        must be in adata_ref.obs.keys(), annotation to project
    annotation_trg : string, optional
        prefix to names of annotations to insert in adata
    annotation_trg_prob : string, optional
        name of best probability annation to insert
    cell_filter_trg : array of boolean, optional
        cell filter forthe  adata object use to subset cell that needs to be projected
    cell_filter_src : arrat of boolean, optional
        cell filter for the adata_ref object use to subset cell that are used to define logistic models and cosine distace reference centroids
    use_variable_genes : boolean (default: True)
        filter genes that are not Variable using scanpy.pp.highly_variable_genes
    use_variable_genes_within_subset : boolead (default : True)
        if true, recomputes variable gene within subset defined by cell_filter_src
    cc_gene_filter_prefix : string (default : None)
        if provided, filters 100 genes associated with cellcycles associated to cell-cycle in human, the string is used as a prefix, so "" can be used is the gene names in adata_ref has no prefix to enable this filter
    use_raw_data : boolean (default : True)
        if True, the input data used for the logistic projection is renormalized from the raw data attribute within adata and adata_ref
    filtered_label : string, optional
        if a cell is filtered by cell_filter_trg, label used to state the cell are not evaluation by the logistic model
    genespeficityfraction : double, optional
        Filters genes whose expression is not celltype specific in the source object, based on 3 heuristic values (fold-increase, fold-increse * fraction-positive, fold-increase * sqrt(fraction=positive)), the top genes with the highest ranks for any of the 3 heuristic value, where the resulting proportion of genes retained matches this input parameter.
    make_circular_coords : boolean (default : False)
        adds a 2D representation of cosine and logistic projection as polar coordinate scattering
    cosine_coor_softmax_coef: double, optional
        coeffient used in softmax transformation of coordinates (default: squarred cosine norm)
    logistic_coor_softmax_coef: double, optional
        coeffient used in softmax transformation of coordinates (default: no transformation used)
    C_parameter : double, array of doubles (default : 1)
        list of regularization parameter used (or single value) for 1 or more logistic projection model training
    doinspect : bool, optional
        A flag used to print this function code before running (default is False)

    Returns
    -------
    adata : {"adata": Anndata.anndata, C_parameter[0] : np.array([nbcell, nbclass]), C_paramtere[1] : ...}
        Modifies adata with additionnal annotations, and produces the full posterior probability rendered by logistic models.
    """   
    if doinspect is True: print("\033[35;46;1mLogisitic regression based projection\033[0m\033[34m"); print(inspect.getsource(doLogisticRegression));print("\033[31;43;1mExecution:\033[0m")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm.libsvm import predict_proba

    if cell_filter_src is not None:
        if isinstance(cell_filter_src, np.ndarray) is False:
            cell_filter_src = np.array(cell_filter_src)
    if cell_filter_trg is not None:
        if isinstance(cell_filter_trg, np.ndarray) is False:
            cell_filter_trg = np.array(cell_filter_trg)

    if annotation_trg is None: annotation_trg = "logist_proj_" + annotation_src #argument function default value
    if annotation_trg_prob is None: annotation_trg_prob = annotation_trg + "_probability"
    if use_variable_genes is True:
        if use_variable_genes_within_subset is True and cell_filter_src is not None:
            print("Computing Variable genes within subset")
            adata_train = anndata.AnnData(adata_ref.raw.X[cell_filter_src,:],obs=adata_ref.obs[cell_filter_src],var=adata_ref.var)
            sc.pp.filter_genes(adata_train, min_cells=5)
            sc.pp.normalize_per_cell(adata_train, counts_per_cell_after=1e4)
            sc.pp.log1p(adata_train)
            sc.pp.highly_variable_genes(adata_train,inplace=True)
            common_genes = sorted(list(set(adata.var_names) & set(adata_train.var_names[adata_train.var["highly_variable"]])))
        else:
            common_genes = sorted(list(set(adata.var_names) & set(adata_ref.var_names[adata_ref.var["highly_variable"]])))
    else:
        common_genes = sorted(list(set(adata.var_names) & set(adata_ref.var_names)))
   
    if cc_gene_filter_prefix is not None:
        print("excluding prior cell cycle genes from " + str(len(common_genes)) + " variable genes")
        common_genes = list( set(common_genes).difference(set([cc_gene_filter_prefix + x for x in ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA","MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]])))

    if genespeficityfraction is not None:
        # heuristic to filter non-informative genes based on class-specific fold change and dropout rate
        print("Filtering gene heuristic")
        if cell_filter_src is None: adata_train = anndata.AnnData(adata_ref.raw.X,obs=adata_ref.obs,var=adata_ref.var)
        else: adata_train = anndata.AnnData(adata_ref.raw.X[cell_filter_src,:],obs=adata_ref.obs[cell_filter_src],var=adata_ref.var)
        adata_train = adata_train[:, common_genes]

        foldchange = np.zeros([len(common_genes),len(adata_train.obs[annotation_src].cat.categories)])
        fractionpositive = np.zeros([len(common_genes),len(adata_train.obs[annotation_src].cat.categories)])
        average = (np.sum(adata_train.X,axis=0) / adata_train.X.shape[0]).tolist()[0]
        nonzero = (np.sum(adata_train.X != 0,axis=0) ).tolist()[0]
        for i in range(len(adata_train.obs[annotation_src].cat.categories)):
            nbcell = sum(adata_train.obs[annotation_src].cat.codes == i)
            foldchange[:,i] = (np.sum(adata_train.X[np.array(adata_train.obs[annotation_src].cat.codes == i),:],axis=0) / nbcell) / average
            fractionpositive[:,i]  = np.sum(adata_train.X[np.array(adata_train.obs[annotation_src].cat.codes == i),:] != 0,axis=0) / nonzero
        from scipy.stats import rankdata
        rankmatrix = np.zeros([len(common_genes),3])
        rankmatrix[:,0] = rankdata(np.amax(foldchange, 1))
        rankmatrix[:,1] = rankdata(np.amax(foldchange * fractionpositive , 1))
        for row in range(len(fractionpositive)):
            for col in range(len(fractionpositive[0])):
                fractionpositive[row][col] = math.sqrt(fractionpositive[row][col])
        rankmatrix[:,2] = rankdata(np.amax(foldchange * fractionpositive , 1))
        maxrank = rankdata(np.amax(rankmatrix, 1))
        common_genes = np.array(common_genes,dtype=object)[maxrank > len(rankmatrix) * (1.0 - genespeficityfraction)]
        print("filtered down to " + str(len(common_genes)) + " genes")

    print("Subsetting objects using " + str(len(common_genes)) + " genes")
    if use_raw_data is True:
        if cell_filter_src is None: adata_train = anndata.AnnData(adata_ref.raw.X,obs=adata_ref.obs,var=adata_ref.var)
        else: adata_train = anndata.AnnData(adata_ref.raw.X[cell_filter_src,:],obs=adata_ref.obs[cell_filter_src],var=adata_ref.var)
        print("Normalizing from raw data")
        
        if cell_filter_trg is None: adata_test = anndata.AnnData(adata.raw.X.copy(), obs= adata.obs,var=adata.var)
        else: adata_test = anndata.AnnData(adata.raw.X[cell_filter_trg,:], obs= adata.obs[cell_filter_trg],var=adata.var)
        if use_max_normalization is True:
            sc.pp.log1p(adata_train)
            sc.pp.log1p(adata_test)
            damax = np.amax(adata_train.X, axis = 1).tocsc()
            for i in range(adata_train.X.shape[0]):
                adata.X.data[adata_train.X.indptr[i]:adata_train.X.indptr[i+1]] / damax[i,0]
            damax = np.amax(adata_test.X, axis = 1).tocsc()
            for i in range(adata_test.X.shape[0]):
                adata_test.X.data[adata_test.X.indptr[i]:adata_test.X.indptr[i+1]] / damax[i,0]
            #adata_train.X = adata_train.X / np.amax(adata_train.X, axis = 1)
            #adata_test.X = adata_test.X / np.amax(adata_test.X, axis = 1)
        else:
            #sc.pp.normalize_per_cell(adata_train, counts_per_cell_after=1e4)
            sc.pp.log1p(adata_train)
            #sc.pp.normalize_per_cell(adata_test, counts_per_cell_after=1e4)
            sc.pp.log1p(adata_test)
        adata_train = adata_train[:, common_genes]
        adata_test = adata_test[:, common_genes]
    else:
        if cell_filter_src is None: adata_train = adata_ref[:,common_genes]
        else: adata_train = adata_ref[cell_filter_src,common_genes]
        if cell_filter_trg is None: adata_test = adata[:,common_genes]
        else: adata_test = adata[cell_filter_trg,common_genes]
    print("training set has " +str(len(adata_train.obs_names)) + " cells and test set has " + str(len(adata_test.obs_names)) + " cells")    
    

    if isinstance(adata_ref.obs[annotation_src].dtype, pd.core.dtypes.dtypes.CategoricalDtype): cl = adata_ref.obs[annotation_src].cat.categories.tolist()
    else: cl = list(set(adata_ref.obs[annotation_src]))
    catlist = np.array(cl)
    tmp = set(adata_train.obs[annotation_src])
    cl = np.array(cl)[ [x in tmp for x in cl] ]
    del tmp

    if do_compute_cosine_distance is True:
        print("Computing Centroids for Cosine distances")
        
        if isinstance(adata_train.obs[annotation_src].dtype, pd.core.dtypes.dtypes.CategoricalDtype): cl = adata_train.obs[annotation_src].cat.categories.tolist()
        else: cl = list(set(adata_train.obs[annotation_src]))



        logcentroid = np.zeros([len(common_genes),len(cl)])
        centroid = np.zeros([len(common_genes),len(cl)])

        adata_mat = anndata.AnnData(adata_ref.raw.X, obs= adata_ref.obs,var=adata_ref.var)[:,common_genes].X # reorders genes
        if cell_filter_src is not None: adata_mat = adata_mat[cell_filter_src,:]
        adata_lmat = np.log1p(adata_mat)
        for i in range(len(cl)):
            flt = np.array(adata_train.obs[annotation_src] == cl[i])
            centroid[:,i] = np.sum(adata_mat[flt,], axis = 0)
            logcentroid[:,i] = np.sum(adata_lmat[flt,], axis = 0)
        centroid = centroid / np.linalg.norm(centroid,axis = 0)
        logcentroid = logcentroid / np.linalg.norm(logcentroid,axis = 0)
        del adata_lmat
        maxindex = np.zeros([adata.raw.X.shape[0]],dtype=int)
        maxcosine = np.zeros([adata.raw.X.shape[0]])
        fullcosine = np.zeros([adata.raw.X.shape[0], centroid.shape[1]])
        maxindexlog = np.zeros([adata.raw.X.shape[0]], dtype=int)
        maxcosinelog = np.zeros([adata.raw.X.shape[0]])
        fullcosinelog = np.zeros([adata.raw.X.shape[0], centroid.shape[1]])
        if make_circular_coords is True:
            logcoor = np.zeros([adata.raw.X.shape[0], 2 ])
            coor = np.zeros([adata.raw.X.shape[0], 2 ])
            projmatrix = np.zeros([2, centroid.shape[1]])
            for i in range(centroid.shape[1]):
                projmatrix[0,i] = math.sin( math.pi * 2 * (i + 0.5) / centroid.shape[1] )
                projmatrix[1,i] = math.cos( math.pi * 2 * (i + 0.5) / centroid.shape[1] )

        c = 0;
        # compute projection, all hail python zen
        print("Compute Cosine distances")

        adata_mat = anndata.AnnData(adata.raw.X, obs= adata.obs,var=adata.var)[:,common_genes].X
        proj = np.zeros(centroid.shape[1])
        lproj = np.zeros(centroid.shape[1])
        sumsqr=0; lsumsqr=0;
        for i in range(len(adata_mat.data)):
            while (i == adata_mat.indptr[c+1]):
                maxcosine[c] = np.amax(proj) / math.sqrt(sumsqr)
                maxindex[c] = np.argmax(proj)
                fullcosine[c,:] = proj / math.sqrt(sumsqr)
                maxcosinelog[c] = np.amax(lproj) / math.sqrt(lsumsqr)
                maxindexlog[c] = np.argmax(lproj)
                fullcosinelog[c,:] = lproj / math.sqrt(lsumsqr)
                if make_circular_coords is True:
                    if cosine_coor_softmax_coef is None:
                        proj = proj * proj
                        lproj = lproj * lproj
                    else:
                        proj =  np.exp(proj * cosine_coor_softmax_coef / math.sqrt(sumsqr))
                        lproj = np.exp(lproj * cosine_coor_softmax_coef / math.sqrt(lsumsqr))
                    if sum(proj) > 0: proj = proj / sum(proj)
                    if sum(lproj) > 0: lproj = lproj / sum(lproj)
                    coor[c,:] = np.matmul(projmatrix, proj) 
                    logcoor[c,:] = np.matmul(projmatrix, lproj)
                c = c + 1
                sumsqr=0;lsumsqr=0;
                proj = np.zeros(centroid.shape[1])
                lproj = np.zeros(centroid.shape[1])
            proj += centroid[adata_mat.indices[i],:] * adata_mat.data[i]
            tmp = np.log1p(adata_mat.data[i])
            lproj += logcentroid[adata_mat.indices[i],:] * tmp
            sumsqr += adata_mat.data[i] * adata_mat.data[i];
            lsumsqr += tmp * tmp
        del adata_mat
        maxcosine[c] = np.amax(proj) / math.sqrt(sumsqr)
        maxindex[c] = np.argmax(proj) 
        maxcosinelog[c] = np.amax(lproj) / math.sqrt(lsumsqr)
        maxindexlog[c] = np.argmax(lproj)
        if make_circular_coords is True:
            if cosine_coor_softmax_coef is None:
                proj = proj * proj
                lproj = lproj * lproj
            else:
                proj =  np.exp(proj * cosine_coor_softmax_coef / math.sqrt(sumsqr))
                lproj = np.exp(lproj * cosine_coor_softmax_coef / math.sqrt(lsumsqr))
            if sum(proj) > 0: proj = proj / sum(proj)
            if sum(lproj) > 0: lproj = lproj / sum(lproj)
            coor[c,:] = np.matmul(projmatrix, proj)
            logcoor[c,:] = np.matmul(projmatrix, lproj)
            adata.obsm["X_cosproj_" + annotation_src] = coor
            adata.obsm["X_lxfcosproj_" + annotation_src] = logcoor
        adata.obsm["cosproj_" + annotation_src] = fullcosine 
        adata.obsm["lxfcosproj_" + annotation_src] = fullcosinelog

        annot = [cl[x] for x in maxindex]
        cflt = [x in set(cl) for x in catlist]
        adata.obs["cosine_proj_" + annotation_src] = pd.Categorical(annot, catlist[cflt], ordered=True)
        adata.obs["cosine_proj_" + annotation_src + "_distance"] = pd.to_numeric(maxcosine, downcast='float')
        if (annotation_src + "_colors") in adata_ref.uns:
            adata.uns["cosine_proj_" + annotation_src + "_colors"] = adata_ref.uns[annotation_src + "_colors"][cflt]
        annot = [cl[x] for x in maxindexlog]
        cflt = [x in set(cl) for x in catlist]
        adata.obs["cosine_logXformedproj_" + annotation_src] = pd.Categorical(annot, catlist[cflt], ordered=True)
        adata.obs["cosine_logXformedproj_" + annotation_src + "_distance"] = pd.to_numeric(maxcosinelog, downcast='float')
        if (annotation_src + "_colors") in adata_ref.uns:
            adata.uns["cosine_logXformedproj_" + annotation_src + "_colors"] = adata_ref.uns[annotation_src + "_colors"][cflt]

    if isinstance(C_parameter, list) is False: C_parameter = [C_parameter]
    for cpar in C_parameter:
    #cpar = C_parameter
        print("Learning classes")
        logisticRegr = LogisticRegression(max_iter = regr_max_iter, n_jobs = 1, random_state = 0, C=cpar,multi_class= "multinomial")
        logisticRegr.fit(adata_train.X, adata_train.obs[annotation_src])
        print("Projecting labels")
        predictions = logisticRegr.predict(adata_test.X)
        probabilities = logisticRegr.predict_proba(adata_test.X)
        
        #    return({"names" : adata_test.obs_names, "predic" : predictions, "probabilities" : probabilities, "model" : logisticRegr})
        print("Storing results")
        probclass = np.full(len(adata.obs_names), filtered_label , dtype= "object")
        probproj = np.zeros(len(adata.obs_names))
        fullmat = np.zeros([len(adata.obs_names), len(cl)])
        dalist = list(adata.obs_names)
        tmap = {}
        for i in range(len(adata.obs_names)):
            tmap.update( {adata.obs_names[i] : i})
        map = [tmap[s] for s in adata_test.obs_names]
    
        if make_circular_coords is True:
            coor = np.zeros([adata.raw.X.shape[0], 2 ])
            projmatrix = np.zeros([2, len(cl)])
            for i in range(len(cl)):
                projmatrix[0,i] = math.sin( math.pi * 2 * (cl.index(logisticRegr.classes_[i]) + 0.5) / len(cl) )
                projmatrix[1,i] = math.cos( math.pi * 2 * (cl.index(logisticRegr.classes_[i]) + 0.5) / len(cl) )
            for i in range(len(map)):
                probclass[map[i]] = predictions[i]
                probproj[map[i]] = np.amax(probabilities[i,:])
                fullmat[map[i],:] = probabilities[i,:]
                if logistic_coor_softmax_coef is not None:
                     probabilities[i,:] = predict_proba[i,:]
    #                probabilities[i,:] = np.exp(probabilities[i,:] * logistic_coor_softmax_coef)
    #                probabilities[i,:] = probabilities[i,:] / np.sum(probabilities[i,:])
                coor[map[i],:] = np.matmul(projmatrix, probabilities[i,:])
    #        if logistic_coor_softmax_coef is not None:
    #            coor = coor * (1.0 + math.exp(-logistic_coor_softmax_coef) *(len(cl) -1))
            adata.obsm["X_logistproj_" + annotation_src + "_" + str(cpar)] = coor
        else:
            for i in range(len(map)):
                probclass[map[i]] = predictions[i]
                probproj[map[i]] = np.amax(probabilities[i,:])

        fullmat2 = np.zeros([len(adata.obs_names), len(cl)])
        for i in range(len(cl)):
            fullmat2[:, cl.index(logisticRegr.classes_[i])] = fullmat[:, i]
        
        cflt = [x in set(probclass) for x in catlist]
        adata.obs[annotation_trg + "_" + str(cpar)] = pd.Categorical(probclass, catlist[cflt], ordered=True)
        adata.obs[annotation_trg_prob + "_" + str(cpar)] = pd.to_numeric(probproj, downcast='float')
        adata.obsm[annotation_trg + "_logist_" + str(cpar)] = fullmat2

    
    if (annotation_src + "_colors") in adata_ref.uns:
        adata.uns[annotation_trg + "_colors"] = adata_ref.uns[annotation_src + "_colors"][cflt]
    return(adata)

def cosineMapping(adata, adatas, ref_obskey, trg_obskey_prefix, ref_cellfilter = None, doinspect=False):
    if doinspect is True: print("\033[35;46;1mCompute Cosine distances\033[0m\033[34m"); print(inspect.getsource(cosineMapping));print("\033[31;43;1mExecution:\033[0m")
    #adata.uns[trg_obskey + "_colors"] = adata_ref.uns[ref_obskey + "_colors"]
    #adatas = splitAnnData(adata_ref, ref_obskey, entryfilter = ref_cellfilter, use_raw_slot_instead=True,getnames=True)
    # match gene names across objects
    print("defining permutation")
    dalist = list(adata.var_names)
    tmap = {}
    for i in range(len(adata.var_names)):
        tmap.update( {adata.var_names[i] : i})
    map = []
    for i in adatas["datalist"][0].var_names:
        if i in tmap:
            map.append(tmap[i])
        else:
            map.append(None)
    # define centroids
    centroid = np.zeros([adata.raw.X.shape[1],len(adatas["datalist"])])
    for i in range(len(adatas["datalist"])):
        davec = np.sum(adatas["datalist"][i].X, axis = 0)
        for j in range(adata.raw.X.shape[1]):
            if map[j] is None:
                centroid[j,i] = 0
            else:
                centroid[j,i] = davec[0,map[j]]
        centroid[:,i] /= math.sqrt(sum(centroid[:,i] * centroid[:,i]))
    maxindex = np.zeros([adata.raw.X.shape[0]])
    maxcosine = np.zeros([adata.raw.X.shape[0]])
    # compute projection ourselves, since python zen means clever code is disallowed to exist
    c = 0;
    proj = np.zeros(centroid.shape[1])
    sumsqr=0;
    for i in range(len(adata.raw.X.data)):
        while (i == adata.raw.X.indptr[c+1]):
            proj /= math.sqrt(sumsqr)
            maxcosine[c] = max(proj)
            aslist = proj.tolist()
            maxindex[c] = aslist.index(maxcosine[c])
            c = c + 1
            sumsqr=0
            proj = np.zeros(centroid.shape[1])
        proj += centroid[adata.raw.X.indices[i],:] * adata.raw.X.data[i] 
        sumsqr += adata.raw.X.data[i] * adata.raw.X.data[i];
    proj /= sqrt(sumsqr)
    maxcosine[c] = max(proj)
    aslist = proj.tolist()
    maxindex[c] = aslist.index(maxcosine[c])
    return({"index" : maxindex , "cosine": maxcosine}) 

def makeValueHeatmap(adata, genes, colG, cellfilter = None, save=None, display_value= "mean", return_count_instead=False, display_gene_names = None, doinspect=False):
    """Makes heatmap

    Parameters
    ----------
    adata : Annadata.anndata
        Object containning the annotations, if the object is Anndata, uses colors from its uns slot (with "_colors" suffix)
    genes : string or list of strings
        list of genes or single key found within adata.obsm.keys()
    save : string, optional
        filename to which the figure is saved
    display_value : "mean", "stddev", "varcoef", "argmax", "argmax_proportion" (default = "mean")
        Selects value to display on heatmap. "varcoef" is mean divided by std. deviation. "argmax" count the number of entries for which the value is the highest, similary "argmax_proportion" is the proportion with the matching class
    return_count_instead : boolean (default = false)
        instead of returning a graph object, return the matrix of values used in the heatmap rendered
    doinspect : bool, optional
        A flag used to print this function code before running (default is False)

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    if doinspect is True: print("\033[35;46;1mMakes an heatmap with mean of values accross a given annotation\033[0m\033[34m"); print(inspect.getsource(makeValueHeatmap));print("\033[31;43;1mExecution:\033[0m")
    
    if isinstance(genes, list): 
        data = adata.X[ [adata.var_names.index(x) for x in genes] ,:]
        if display_gene_names is None: display_gene_names = genes
    else: data = adata.obsm[genes]


    if isinstance(adata.obs[colG].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
        setG = np.array(adata.obs[colG].cat.categories, dtype= object)
        tmpset = np.array(list(set(adata.obs[colG].cat.codes)), dtype = int)
        setG = setG[np.array([int(x) for x in tmpset], dtype = int)]
        map = {tmpset[x] : x for x in range(len(setG))}
        indexG = np.array([ int(map[x]) for x in adata.obs[colG].cat.codes], dtype='int')
    else:
        setG = list(set(adata.obs[colG]))
        indexG = len(set(adata.obs[colG]))
   
    meanvalue = np.zeros( [len(setG) , data.shape[1]] )
    if display_value in ["argmax_proportion" , "argmax"]:
        maxindex = np.argmax(data,axis =1)
        for x in range(len(setG)):
            for y in range(data.shape[1]):
                meanvalue[x,y] = np.sum(maxindex[(indexG == x)] == y)
        if display_value == "argmax_proportion":
            for x in range(len(setG)):
                meanvalue[x,:] = meanvalue[x,:] / sum(indexG == x)
    elif display_value == "stddev":
        for x in range(len(setG)):
            nb = sum(indexG == x)
            mean = np.sum(data[(indexG == x),:],axis=0).reshape([-1])
            sqred = np.sum(data[(indexG == x),:] * data[(indexG == x),:],axis=0).reshape([-1])
            meanvalue[x,:] = np.sqrt(sqred * nb - mean * mean) / nb
    elif display_value == "varcoef":
        for x in range(len(setG)):
            nb = sum(indexG == x)
            mean = np.sum(data[(indexG == x),:],axis=0).reshape([-1])
            sqred = np.sum(data[(indexG == x),:] * data[(indexG == x),:],axis=0).reshape([-1])
            meanvalue[x,:] = mean / np.sqrt(sqred * nb - mean * mean)
    else:
        for x in range(len(setG)):
            meanvalue[x,:] = np.sum(data[(indexG == x),:],axis=0).reshape([-1]) / sum(indexG == x)

    if (return_count_instead) :  return(pd.DataFrame(meanvalue ,  index = setG, columns = display_gene_names))
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    plt.colorbar(plt.imshow(meanvalue ))
    ax.set_yticklabels(setG)
    ax.set_yticks(np.arange(len(setG)) )
    ax.set_xticklabels(display_gene_names, rotation = 90)
    ax.set_xticks(np.arange(len(display_gene_names)) )
    if save is not None:
        plt.savefig( "figures/" + save)
    return(ax)

def makeBarplot(df, colC, colX, colG =None, flt = None, color = None, on_graph_labels = None, return_count_instead=False, makeProportion = True, plotattrdico= {"yaxe_title" : None}, save= None):
    """Make a bar graph or a stacked bar graph from frenquencies from a given annotation pair or tripplet, respectively.

    Parameters
    ----------
    df : Annadata.anndata or panda.dataframe
        Object containning the annotations, if the object is Anndata, uses colors from its uns slot (with "_colors" suffix)
    on_graph_labels : dictionnary with (tuple<string, string, string> : tuple<string, string>) entries
        list of labels to insert on the figure, relative to each bar segement inserted. The dictorary keys identify the rect bar uniquely (color Annotation, X-axis annotation, Group annotation) (using the annotations 
    doinspect : bool, optional
        A flag used to print this function code before running (default is False)

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    if isinstance(df, anndata._core.anndata.AnnData):
        if colC + "_colors" in df.uns.keys():
            color =  df.uns[colC + "_colors"]
        df = df.obs
    if flt is not None: 
        df = df[flt]
    if isinstance(df[colX].dtype, pd.core.dtypes.dtypes.CategoricalDtype): 
        setX = np.array(df[colX].cat.categories, dtype= object)
        tmpset = np.array(list(set(df[colX].cat.codes)), dtype = int)
        setX = setX[np.array([int(x) for x in tmpset], dtype = int)]
        map = {tmpset[x] : x for x in range(len(setX))}
        indexX = [ int(map[x]) for x in df[colX].cat.codes]
    else: setX = list(set(df[colX])) # TODOINDEX
    if isinstance(df[colC].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
        setC = np.array(df[colC].cat.categories, dtype= object)
        tmpset = np.array(list(set(df[colC].cat.codes)), dtype = int)
        setC = setC[np.array([int(x) for x in tmpset], dtype = int)]
        map = {tmpset[x] : x for x in range(len(setC))}
        indexC = [ int(map[x]) for x in df[colC].cat.codes]
    else: setC = list(set(df[colC])) # TODO index

    if colG is None: 
        setG = ["data"]
        indexG = np.zeros(df.shape[0])
    else: 
        if isinstance(df[colG].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            setG = np.array(df[colG].cat.categories, dtype= object)
            tmpset = np.array(list(set(df[colG].cat.codes)), dtype = int)
            setG = setG[np.array([int(x) for x in tmpset], dtype = int)]
            map = {tmpset[x] : x for x in range(len(setG))}
            indexG = [ int(map[x]) for x in df[colG].cat.codes]
        else:
            setG = list(set(df[colG]))
            indexG = len(set(df[colG])) # TODO
    counts = np.zeros([len(setG) , len(setC) , len(setX)])
    for x in range(df.shape[0]):
        counts[indexG[x],indexC[x],indexX[x]]+=1

    if (return_count_instead is True): return counts

    X = np.arange(len(setX))
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    
    baseheight = np.zeros(len(setX))
    if color is None:
        color = ["#000000", "#888888","#000000", "#888888","#000000", "#888888","#000000", "#888888","#000000", "#888888","#000000", "#888888"]

    if makeProportion is True: 
        denum = np.sum(counts,axis=1)
        denum[denum == 0] = 1
    else: denum = np.full([len(setG) , len(setX)], 1)

    if (return_count_instead is True): 
        if makeProportion is True:
            for y in range(len(setG)):
                for x in range(len(setX)):
                    counts[y, :, x] = counts[y, :, x] / denum[y,x]
        return counts
    
    for g in range(len(setG)):
        baseheight = np.zeros(len(setX))
        for c in range(len(setC)):
            ax.bar(X + ((1 +g) / (1 + len(setG))), counts[g,c,:] / denum[g,:] , bottom= baseheight, color = color[c], width = 1.0 / (1 + len(setG)))
            if on_graph_labels is not None:
                for x in range(len(setX)):
                    curkey = tuple([setC[c], setX[x], setG[g]])
                    if curkey in on_graph_labels.keys():
                        print((counts[g,c,x] / denum[g,x])  + baseheight[x])
                        plt.text(x + (g +1) / (1 + len(setG)), (counts[g,c,x] / denum[g,x]) + baseheight[x] , on_graph_labels[curkey][0],horizontalalignment = 'center')
            baseheight += counts[g,c,:] / denum[g,:]
    ax.legend(labels=setC, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xticks(np.arange(len(setX)) + 0.5)
    ax.set_xticklabels(setX, rotation = 90)
    import matplotlib.ticker as mtick
   # ax.yaxis.set_major_formatter( mtick.FormatStrFormatter("%.0f%%"))
    if "yaxe_title" in plotattrdico.keys():
        if plotattrdico["yaxe_title"] is not None: ax.set_ylabel(plotattrdico["yaxe_title"])
        elif makeProportion is True: ax.set_ylabel("Percentage")
        else: ax.set_ylabel("Count")
    if "xaxe_title" in plotattrdico.keys(): ax.set_xlabel(plotattrdico["xaxe_title"])
    if makeProportion is True:  ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))
    if save is not None:
        plt.savefig( "figures/" + save)
#(scanpy=True, dpi=80, dpi_save=150, frameon=True, vector_friendly=True, fontsize=14, figsize=None, color_map=None, format='pdf', facecolor=None, transparent=False, ipython_format='png2x'

    return counts

def makeWheelPlot(adata, coor_annot, color_annot, tiplabels, angle_adjust =0, pt_size = 5, save= None, doinspect=False):
    """Make a Wheel plot using precomputed coordinates in AnnData object

    Parameters
    ----------
    adata : Annadata.anndata
        Object containing the annotations and coordinates to overlay
    doinspect : bool, optional
        A flag used to print this function code before running (default is False)

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    if doinspect is True: print("\033[35;46;1mCompute Cosine distances\033[0m\033[34m"); print(inspect.getsource(makeWheelPlot));print("\033[31;43;1mExecution:\033[0m")
    
    if isinstance(adata.obs[color_annot].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
        setC = np.array(adata.obs[color_annot].cat.categories, dtype= object)
        tmpset = np.array(list(set(adata.obs[color_annot].cat.codes)), dtype = int)
        setC = setC[np.array([int(x) for x in tmpset], dtype = int)]
        map = {tmpset[x] : x for x in range(len(setC))}
        indexC = [ int(map[x]) for x in adata.obs[color_annot].cat.codes]
    else: setC = list(set(adata.obs[color_annot])) # TODO index    

    colannot = "cosine_logXformedproj_bulkorg"
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    ax.scatter(adata.obsm[coor_annot][:,0], adata.obsm[coor_annot][:,1], c=adata.obs[color_annot].cat.codes,
        cmap=LinearSegmentedColormap.from_list('my_cm',adata.uns[color_annot + "_colors"], N=len(adata.uns[color_annot + "_colors"])), edgecolors='none', s=pt_size)
    n = len(tiplabels)
    x1, y1 = [math.sin(angle_adjust + math.pi * 2 * (x + 0.5) / n) for x in range(n+1) ], [ math.cos(angle_adjust + math.pi * 2 * (x + 0.5) / n) for x in range(n+1)]
    plt.plot(x1, y1,marker = '', color = "#444444")
    for i in range(n):
        if (y1[i] > 0): 
            if (x1[i] > 0):  plt.annotate(tiplabels[i], (x1[i],y1[i]+0.05), ha = "left",textcoords="offset points")
            else: plt.annotate(tiplabels[i], (x1[i],y1[i]+0.05), ha = "right",textcoords="offset points")
        else: 
            if (x1[i] > 0):  plt.annotate(tiplabels[i], (x1[i],y1[i]-0.1), ha = "left",textcoords="offset points")
            else: plt.annotate(tiplabels[i], (x1[i],y1[i]-0.1), ha = "right",textcoords="offset points")
    plt.axis('off')
    plt.xlim([-1.5, 1.8]); plt.ylim([-1.2, 1.2])


  #  from matplotlib.lines import Line2D
   # custom_legend = [Line2D([0], [0], color=adata.uns[color_annot + "_colors"][i], label='Scatter',marker='s') for i in range(len(setC))]
    from matplotlib.patches import Patch
    custom_legend = [Patch(facecolor=adata.uns[color_annot + "_colors"][i], edgecolor=adata.uns[color_annot + "_colors"][i])   for i in range(len(setC))]

    ax.legend(custom_legend, setC, bbox_to_anchor=(1.05, 1), loc='upper left')
    if save is not None:
        plt.savefig( "figures/" + save)
    return ax



