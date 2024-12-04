#### Import ComplexHeatmap
library("ComplexHeatmap")


#### Matt's plotting functions 

#' Load the fitted exposures and normalise them
#'
#' Load's the exposure table, separates out the goodness-of-fit metrics and exposures and normalises the exposures to sum to 1 across each sample.
#'
#' @param tgt The relevant *_fitExposures.tsv file that results should be loaded from.  Can either be the exact file, or the base part to which _fitExposures.tsv will be added.
#' @return A list containing the normalised exposure table, raw exposure table, and a goodness-of-fit table
normaliseExposures = function(tgt){
  #Is this just the base?
  if(file.exists(paste0(tgt,'_fitExposures.tsv')))
    tgt = paste0(tgt,'_fitExposures.tsv')
  fit = read.table(tgt,sep='\t',header=TRUE)
  #Extract the goodness of fit rows
  gofNoms = c('pR2','fitCount','obsCount')
  gof = fit[gofNoms,]
  gof['log2(countRatio)',] = log2(unlist(gof['fitCount',]/gof['obsCount',]))
  #And the exposure table
  exposures = fit[-match(gofNoms,rownames(fit)),]
  #Normalise
  exposures = t(t(exposures)/colSums(exposures))
  #Done
  return(list(exposures=exposures,gof=gof,raw=fit[-match(gofNoms,rownames(fit)),]))
}

#' Plot normalised exposures
#' 
#' Takes the normalised exposure object produced by \code{normaliseExposures} and creates a basic heatmap to visualise the results.
#'
#' @param fit The normalised exposure object returned by \code{normaliseExposures}
#' @param exposureScale Range of exposures to show.
#' @param cluster_rows Should we cluster the rows.
#' @param ... Passed to Heatmap.
plotExposures = function(fit,exposureScale=c(0,0.5),cluster_rows=FALSE,use_raster = FALSE){
  #Colours for exposures
  hmCols = c('#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000')
  #Colours for pR2 metric
  pR2Cols  = c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')
  #Colours for log library sizes
  libCols = c('#fcfbfd','#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#54278f','#3f007d')
  #And library size ratio
  libRatCols = c('#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e')
  #Create the bottom annotation
  gof = fit$gof
  gof['fitCount',] = log10(gof['fitCount',])
  gof['obsCount',] = log10(gof['obsCount',])
  rownames(gof) = gsub('(.*Count)','log10(\\1)',rownames(gof))
  #Decide on range for library size
  libRange = range(gof[grep('Count',rownames(gof)),])
  #Convert colours to ramps
  bCols = circlize::colorRamp2(seq(0,1,length.out=length(pR2Cols)),pR2Cols)
  lCols = circlize::colorRamp2(seq(libRange[1],libRange[2],length.out=length(libCols)),libCols)
  hmColObj = circlize::colorRamp2(seq(exposureScale[1],exposureScale[2],length.out=length(hmCols)),hmCols)
  lrCols = circlize::colorRamp2(seq(-1,1,length.out=length(libRatCols)),libRatCols)
  botAnno = HeatmapAnnotation(df = t(gof),
                              annotation_name_side = 'left',
                              col = list(pR2 =bCols,
                                         `log10(fitCount)` = lCols,
                                         `log10(obsCount)` = lCols,
                                         `log2(countRatio)` = lrCols)
  )
  hm = Heatmap((fit$exposures),
               col=hmColObj,
               name='Exposures',
               bottom_annotation=botAnno,
               cluster_rows = cluster_rows
  )
  return(hm)
}


#### Visualize results of deconvolution

path_to_results = '/nfs/team292/vl6/CancerDeconvolution/output_ovarianCancer_FRT_BRCA12/'
out = normaliseExposures(paste0(path_to_results, '_fitExposures.tsv'))

# Read in molecular subtype of cancers 
molecular = read.csv2("/nfs/team292/vl6/CancerDeconvolution/ovarianCancer_metadata.csv", sep = ',')
molecular$UniqueSampleID = gsub('-', '.', molecular$UniqueSampleID)
library(dplyr)
molecular = molecular %>% select(c(UniqueSampleID, Subtype_mRNA, cgc_case_clinical_stage, xml_neoplasm_histologic_grade))

#Colours for exposures
hmCols = c('#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000')
#Colours for pR2 metric
pR2Cols  = c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')
#Colours for log library sizes
libCols = c('#fcfbfd','#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#54278f','#3f007d')
#And library size ratio
libRatCols = c('#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e')
#Create the bottom annotation
gof = out$gof
gof['fitCount',] = log10(gof['fitCount',])
gof['obsCount',] = log10(gof['obsCount',])
rownames(gof) = gsub('(.*Count)','log10(\\1)',rownames(gof))
#Decide on range for library size
libRange = range(gof[grep('Count',rownames(gof)),])
#Convert colours to ramps
bCols = circlize::colorRamp2(seq(0,1,length.out=length(pR2Cols)),pR2Cols)
lCols = circlize::colorRamp2(seq(libRange[1],libRange[2],length.out=length(libCols)),libCols)
subtype_cols = c('#4287f5', '#f5b82a', '#f250d2', '#3ebf17', '#944d48')
table(merged$Subtype_mRNA)
sCols = circlize::colorRamp2(c(0, 1, 2, 3, 4), subtype_cols)
namedcolors <- subtype_cols
names(namedcolors) <- c("Differentiated", "Immunoreactive", "Mesenchymal", 'NA', 'Proliferative')

exposureScale=c(0,0.5)
hmColObj = circlize::colorRamp2(seq(exposureScale[1],exposureScale[2],length.out=length(hmCols)),hmCols)
lrCols = circlize::colorRamp2(seq(-1,1,length.out=length(libRatCols)),libRatCols)
ciao = t(gof)
ciao = as.data.frame(ciao)

# Merge molecular dataframe with ciao dataframe 
ciao$UniqueSampleID <- rownames(ciao)
merged <- merge(ciao, molecular ,by="UniqueSampleID", all.x = TRUE)
rownames(merged) <- merged$UniqueSampleID
merged$UniqueSampleID <- NULL
merged <- merged %>% select(c(pR2, `log10(fitCount)`, `log10(obsCount)`, `log2(countRatio)`, Subtype_mRNA))
merged <- merged %>% mutate(Subtype_mRNA = replace_na(Subtype_mRNA, "NA"))

botAnno = HeatmapAnnotation(df = merged,
                            annotation_name_side = 'left',
                            col = list(pR2 =bCols,
                                       `log10(fitCount)` = lCols,
                                       `log10(obsCount)` = lCols,
                                       `log2(countRatio)` = lrCols,
                                       Subtype_mRNA = namedcolors)
)


#Subtype_mRNA = namedcolors

#row.names.remove <- c("Epithelial")
#no_epithelials <- out$exposures[!(row.names(out$exposures) %in% row.names.remove), ]


pdf("/home/jovyan/Adult_FemaleReproductiveTract/figures/deconvolution/FetalAdult_Epithelial_FRT_OvarianCancer_BRCA12.pdf",width=8.75,height=6.75)
Heatmap((out$exposures),
        col=hmColObj,
        bottom_annotation=botAnno,
        name='Exposures',
        row_names_side = "left",
        cluster_rows = FALSE,
        cluster_columns = TRUE,
        show_column_names = FALSE,
        
        column_title = "Ovarian Serous Cystadenocarcinoma TCGA with BRCA1/2 mutations (13 patients)",
        row_title = "scRNAseq Female Reproductive Tract epithelial cell signals (fetal + adult)",
)
dev.off()


######## METADATA ######
path_to_metadata = "/lustre/scratch117/cellgen/cellgeni/team274/bulkData/"
metadata = readRDS(paste0(path_to_metadata, "TCGA.RDS"))

###### RETRIEVE METADATA DIRECTLY FROM TCGA ##########

library(maftools)
library(dplyr)
library(TCGAbiolinks)
maf <- GDCquery(project = "TCGA-OV", 
                data.category = "Simple nucleotide variation", 
                data.type = "Simple somatic mutation",
                access = "open", 
                legacy = TRUE)

# Check maf availables
library(DT)
datatable(dplyr::select(getResults(maf),-contains("cases")),
          filter = 'top',
          options = list(scrollX = TRUE, keys = TRUE, pageLength = 10), 
          rownames = FALSE)

query.maf.hg19 <- GDCquery(project = "TCGA-OV", 
                           data.category = "Simple nucleotide variation", 
                           data.type = "Simple somatic mutation",
                           access = "open", 
                           file.type = "genome.wustl.edu_OV.IlluminaGA_DNASeq.1.3.somatic.maf",
                           legacy = TRUE)
GDCdownload(query.maf.hg19)
maf <- GDCprepare(query.maf.hg19)
write.csv(maf, '/nfs/team292/vl6/CancerDeconvolution/TCGA_OV_genome.wustl.edu_OV.IlluminaGA_DNASeq.1.3.somatic.csv')

maf <- maf %>% read.maf
datatable(getSampleSummary(maf),
          filter = 'top',
          options = list(scrollX = TRUE, keys = TRUE, pageLength = 5), 
          rownames = FALSE)
plotmafSummary(maf = maf, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE)
