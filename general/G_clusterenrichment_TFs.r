#!/usr/bin/env Rscript


args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("please provide the path to the folder with the DEGs. Ex. 'figures/cluster_markers/' ", call.=FALSE)
}else{
    message('DEGs folder: ', args[1])
}


# Set
require(viper)
require(reshape2)
home = '.'
setwd(home)


# Find DEGs files
cl_files = list.files(path = args[1], pattern = '_DEGs.csv', full.names = T)
message(length(cl_files), ' DEGs files')


# Load regulons
viper_gset = get(load('~/farm/gsea/genesets/dorotheav2-top10scoring_VentoLab20200121.rdata'))
#viper_gset = get(load('~/farm/gsea/genesets/trrust_viper.rdata'))
#viper_gset = get(load('~/farm/gsea/genesets/omnipath_TOP_viperRegulon.rdata'))
message('Analysizing ', length(viper_gset), ' TFs')


# For each DEG file
results = list()
for (clf in cl_files){
  cl_name = gsub('.csv', '', tail(unlist(strsplit(clf, split = '/')),1))
  cl_name = paste('cl_', cl_name, sep = '')
  message('\n', cl_name)
  
  DEsignature = read.csv(clf, stringsAsFactors = F)
  # Exclude probes with unknown or duplicated gene symbol
  DEsignature = subset(DEsignature, Gene != "" )
  DEsignature = subset(DEsignature, ! duplicated(Gene))
  # Estimate z-score values for the GES. Cheeck VIPER manual for details
  myStatistics = matrix(DEsignature$logFC, dimnames = list(DEsignature$Gene, 'logFC') )
  myPvalue = matrix(DEsignature$P.Value, dimnames = list(DEsignature$Gene, 'P.Value') )
  mySignature = (qnorm(myPvalue/2, lower.tail = FALSE) * sign(myStatistics))[, 1]
  mySignature = mySignature[order(mySignature, decreasing = T)]
  # Estimate TF activities
#   mrs = msviper(ges = mySignature, regulon = viper_gset, minsize = 4, ges.filter = F)
  mrs = msviper(ges = myStatistics[,1][order(myStatistics[,1], decreasing = T)], regulon = viper_gset, minsize = 4, ges.filter = F)
  cl_enrichment = data.frame(Regulon = names(mrs$es$nes),
                             cl_name = cl_name,
                             Size = mrs$es$size[ names(mrs$es$nes) ], 
                             NES = mrs$es$nes, 
                             p.value = mrs$es$p.value, 
                             FDR = p.adjust(mrs$es$p.value, method = 'fdr'))
  cl_enrichment = subset(cl_enrichment, Size < 200)
  cl_enrichment = subset(cl_enrichment, FDR < 0.05)
  cl_enrichment = cl_enrichment[ order(cl_enrichment$p.value), ]
  if( nrow(cl_enrichment) > 0 )
    results[[cl_name]] = cl_enrichment
}

df = melt(results, id.vars = names(results[[1]]))
df = df[, c(2,1,4:6,3)]
df = df[ order(df$p.value), ]
write.csv(df, file = paste0(args[1], '/TFs_activities.csv'), row.names = F, quote = F)
