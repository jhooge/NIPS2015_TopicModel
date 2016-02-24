library(tm)
library(lda)
library(LDAvis)

library(parallel)

library(data.table)
library(Rmpfr)
library(ggplot2)
library(reshape2)

library(tsne) # low dimensional embedding
library(caret)

library(rbokeh)

source("~/workspace/R/Utilities/HelpersParallelization.R")

papers = fread("~/Datasets/kaggle/NIPS2015_papers/Papers.csv")
docs = papers$Abstract

stop_words <- stopwords("SMART")

# pre-processing:
docs <- gsub("[[:punct:]]", " ", docs)  # replace punctuation with space
docs <- gsub("[[:cntrl:]]", " ", docs)  # replace control characters with space
docs <- gsub("^[[:space:]]+", "", docs) # remove whitespace at beginning of documents
docs <- gsub("[[:space:]]+$", "", docs) # remove whitespace at end of documents
docs <- tolower(docs)  # force to lowercase

# tokenize on space and output as a list:
doc.list <- strsplit(docs, "[[:space:]]+")

# Remove all words with less than 4 characters
doc.list <- lapply(doc.list, function(x) x[sapply(x, nchar)>3])

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)


# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (2,000)
W <- length(vocab)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [8939, 5544, 2411, 2410, 2143, ...]


# MCMC and model tuning parameters:
K <- 20
G <- 5000 # number of iterations?
alpha <- 0.02
eta <- 0.02
ntopics = seq(2, 20, by=1)

# Fit the model:
set.seed(357)

worker <- function() {
  bindToEnv(objNames=c("documents", "vocab",
                       "G", "alpha", "eta"))
  function(k) {
    lda::lda.collapsed.gibbs.sampler(documents = documents, K = k, vocab = vocab, 
                                     num.iterations = G, alpha = alpha, 
                                     eta = eta, initial = NULL, burnin = 0,
                                     compute.log.likelihood = TRUE)
  }
}

t1 <- Sys.time()
cluster <- startCluster()
models <- parLapply(cluster, ntopics, worker())
shutDownCluster(cluster)
t2 <- Sys.time()
t2 - t1  # about 24 minutes on laptop

names(models) <- ntopics

harmonicMean = function(logLikelihoods, precision=2000L) {
  llMed = median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

logLiks = lapply(models, function(L)  L$log.likelihoods)
harmMeans = sapply(logLiks, function(h) harmonicMean(h))
k = which(harmMeans==max(harmMeans))

plot(ntopics, harmMeans, type = "l")
k=8
model = models[[k]]

## Get the top words in the cluster
N <- 5 
top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
top.words.df <- as.data.frame(top.words)
colnames(top.words.df) <- ntopics[1]:ntopics[k]

top.documents <- top.topic.documents(model$document_sums, 
                                     num.documents = 20, 
                                     alpha = alpha)
top.documents.df <- as.data.frame(top.documents)
colnames(top.documents.df) <- ntopics[1]:k


# Get topics with maximum log likelihood for docs
topic.proportions <- t(model$document_sums) / colSums(model$document_sums)
topic.proportions <- topic.proportions + 0.000000001 ## Laplace smoothing

getTopic <- function(topic.vec) {
  argmax <- which(topic.vec==max(topic.vec))
  if(length(argmax)>1){
    argmax <- argmax[1]
  }
  return(argmax)
}

getMaxLogLik <- function(topic.vec) {
  return(topic.vec[getTopic(topic.vec)])
}

# assign each document the topic with maximum probability
doc.topic <- unlist(apply(topic.proportions, 1, getTopic))
doc.topic.words <- apply(top.words[,doc.topic], 2, paste, collapse=".")
doc.logLik <- unlist(apply(topic.proportions, 1, getMaxLogLik))

## Compute Jensen-Shannon Divergence between documents
## p,q probability distribution vectors R^n /in [0,1]
JSD <- function(p, q) {
  m <- 0.5 * (p + q)
  divergence <- 0.5 * (sum(p * log(p / m)) + sum(q * log(q / m)))
  return(divergence)
}

n <- dim(topic.proportions)[1]
X <- matrix(rep(0, n*n), nrow=n, ncol=n)
indexes <- t(combn(1:nrow(topic.proportions), m=2))
for (r in 1:nrow(indexes)) {
  i <- indexes[r, ][1]
  j <- indexes[r, ][2]
  p <- topic.proportions[i, ]
  q <- topic.proportions[j, ]
  X[i, j] <- JSD(p,q) 
}
X <- X+t(X)

## Dimension Reduction of Jensen-Shannon Distance matrix
X_dist <- sqrt(X) # compute Jensen-Shannon Distance

X_MDS_projected <- cmdscale(X_dist, k = 2) ## Multi dimensional scaling
X_tSNE_projected <- tsne(X_dist, k = 2, initial_dims = 10, perplexity = 40) ## t-SNE projection
preProc = preProcess(X_dist, method=c("center", "scale", "pca"))
X_PCA_projected = predict(preProc, X_dist)[,1:2] # PCA projection

projections <- data.frame(Topic=as.factor(doc.topic), 
                          TopWords=as.factor(doc.topic.words),
                          Proportion=doc.logLik,
                          Title=papers$Title,
                          EventType=papers$EventType,
                          Abstract=papers$Abstract,
                          x_pca=X_PCA_projected[, 1], 
                          y_pca=X_PCA_projected[, 2],
                          x_mds=X_MDS_projected[, 1], 
                          y_mds=X_MDS_projected[, 2],
                          x_tsne=X_tSNE_projected[, 1], 
                          y_tsne=X_tSNE_projected[, 2])


tools <- c("pan", 
           "wheel_zoom", "box_zoom", 
           "box_select", "lasso_select", 
           "reset", "save")                            
## PCA Plot
pca_fig <- figure(tools=tools) %>%
  ly_points(x_pca, y_pca, data = projections,
            color = Topic, size = Proportion*10,
            hover = list(TopWords, Proportion, Title, 
                         EventType))
## MDS Plot
mds_fig <- figure(tools=tools) %>%
  ly_points(x_mds, y_mds, data = projections,
            color = Topic, size = Proportion*10,
            hover = list(TopWords, Proportion, Title, 
                         EventType))
## t-SNE Plot
tsne_fig <- figure(tools=tools) %>%
  ly_points(x_tsne, y_tsne, data = projections,
            color = Topic, size = Proportion*10,
            hover = list(TopWords, Proportion, Title, 
                         EventType))

projList <- list(pca_fig, mds_fig, tsne_fig)
p = grid_plot(projList, ncol=2, link_data=TRUE)
p



## For M random documents, display their topic fingerprints
M <- 5
topic.proportions <- topic.proportions[sample(1:dim(topic.proportions)[1], M),]
topic.proportions[is.na(topic.proportions)] <-  1 / K
colnames(topic.proportions) <- apply(top.words, 2, paste, collapse=".")
colnames(topic.proportions) <- 1:9

topic.proportions.df <- as.data.frame(topic.proportions)
molten <- melt(topic.proportions.df)
colnames(molten) <- c("Topic", "Proportion")

d <- data.frame(Topic=colnames(topic.proportions.df), 
                Proportion=as.numeric(topic.proportions.df[2,]))

figure(tools=tools) %>%
  ly_bar(x=Topic, y=Proportion, data = d, color=Topic,
         alpha = 1, position = c("stack", "fill", "dodge"), width = 0.9,
         origin = NULL, breaks = NULL, right = FALSE, binwidth = NULL,
         lname = NULL, lgroup = NULL, legend = NULL)

## If a document has not been assigned to a topic it has the same probability tp belong to any topic
topic.proportions[is.na(topic.proportions)] <-  1 / K
colnames(topic.proportions) <- apply(top.words, 2, paste, collapse=".")

topic.proportions.df <- melt(cbind(data.frame(topic.proportions),
                                   document=factor(1:M)), 
                             variable.name="topic",
                             id.vars = "document") 

qplot(topic, value, fill=document, ylab="proportion",
      data=topic.proportions.df, geom="bar", stat="identity") +
  theme(axis.text.x = element_text(angle=90, hjust=1)) +
  coord_flip() +
  facet_wrap(~ document, ncol=5)

theta <- t(apply(model$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(model$topics) + eta, 2, function(x) x/sum(x)))

modelParams <- list(phi = phi,
                      theta = theta,
                      doc.length = doc.length,
                      vocab = vocab,
                      term.frequency = term.frequency)

# create the JSON object to feed the visualization:
json <- createJSON(phi = modelParams$phi, 
                   theta = modelParams$theta, 
                   doc.length = modelParams$doc.length, 
                   vocab = modelParams$vocab, 
                   term.frequency = modelParams$term.frequency)

serVis(json, open.browser = TRUE)
