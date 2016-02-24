---
title: "Latent Dirichlet Allocation Topic Modeling (lda) on NIPS 2015 Papers"
author: "Jens Hooge"
date: "16. Februar 2016"
output: html_document
---

# Introduction
Topic models are probabilistic latent variable models of documents that exploit the correlations among the words and latent semantic themes” (Blei and Lafferty, 2007). The name "topics" signifies the hidden, to be estimated, variable relations (=distributions) that link words in a vocabulary and their occurrence in documents. A document is seen as a mixture of topics. This intuitive explanation of how documents can be generated is modeled as a stochastic process which is then "reversed"" (Blei and Lafferty, 2009) by machine learning techniques that return estimates of the latent variables. With these estimates it is possible to perform information retrieval or text mining tasks on a document corpus.

# Loading required libraries
In this study we will utilize R tm R package for querying and textmining of the NIPS Papers 2015
```{r, warning=FALSE, message=FALSE}
library(tm) ## texmining
library(lda) ## the actual LDA model
library(LDAvis) # visualization library for LDA

library(parallel) # multi-core paralellization

library(data.table) # fread
library(Rmpfr) # harmonic mean maximization
library(ggplot2) # pretty plotting lib
library(reshape2) # reformatting lib for ggplot2

library(tsne) # low dimensional embedding
library(caret) # ml model wrapper lib, but only used for data transformation here

library(rbokeh) # pretty (interactive) plotting


load("ldaModels_NIPS2015.rda")
load("vocab.rda")
load("termTable.rda")
load("documents.rda")
```

# Helper Functions
```{r}
#' Copy arguments into env and re-bind any function's lexical scope to bindTargetEnv .
#' 
#' See http://winvector.github.io/Parallel/PExample.html for example use.
#' 
#' 
#' Used to send data along with a function in situations such as parallel execution 
#' (when the global environment would not be available).  Typically called within 
#' a function that constructs the worker function to pass to the parallel processes
#' (so we have a nice lexical closure to work with).
#' 
#' @param bindTargetEnv environment to bind to
#' @param objNames additional names to lookup in parent environment and bind
#' @param names of functions to NOT rebind the lexical environments of
bindToEnv <- function(bindTargetEnv=parent.frame(), objNames, doNotRebind=c()) {
  # Bind the values into environment
  # and switch any functions to this environment!
  for(var in objNames) {
    val <- get(var, envir=parent.frame())
    if(is.function(val) && (!(var %in% doNotRebind))) {
      # replace function's lexical environment with our target (DANGEROUS)
      environment(val) <- bindTargetEnv
    }
    # assign object to target environment, only after any possible alteration
    assign(var, val, envir=bindTargetEnv)
  }
}

startCluster <- function(cores=detectCores()) {
  cluster <- makeCluster(cores)
  return(cluster)
}

shutDownCluster <- function(cluster) {
  if(!is.null(cluster)) {
    stopCluster(cluster)
    cluster <- c()
  }
}
```

# Read the NIPS 2015 Corpus

First we will read all the nips papers in a data frame. We will use the abstracts for the training of our LDA
```{r}
papers = fread("~/Datasets/kaggle/NIPS2015_papers/Papers.csv")
docs = papers$Abstract
```

# Preprocessing
To train the LDA in the later steps, we need the word frequencies in each of those abstracts. For representative word frequencies we removed a number of problematic characters, removed punctuation, control characters, whitespaces, stopwords which belonged to the SMART stopword collection, all words with less than 4 characters and words which occurred less than 4 times in the documents. Lastly we transformed each word to lowercase.

```{r, eval=FALSE}
stop_words <- stopwords("SMART")
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

save(vocab, file="vocab.rda")
save(term.table, file="termTable.rda")
```

Next we reformated the documents into the format required by the lda package.
```{r, eval=FALSE}
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

save(documents, file="documents.rda")
```

Before we start training our LDA, we first will calculate some statistics related to the data set:
```{r}
D <- length(documents)  # number of documents
W <- length(vocab)  # number of terms in the vocab
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document
N <- sum(doc.length)  # total number of tokens in the data
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus
```

```{r}
df <- data.frame("Number of Documents"=D, 
                 "Number of Terms in Vocabulary"=W,
                 "Total Number of Tokens in Corpus"=N)

knitr::kable(df, digits = 0, caption = "Document Statistics")
```

# LDA Parameters
Now we can define the parameters for our LDA Model. We will train LDA models assuming between 2 and 10 topics in out document corpus.
TODO: Explain the different Parameters (see notes and http://videolectures.net/mlss09uk_blei_tm/)
TODO: Tune the hyperparameters too, using expand.grid
TODO: Fix harmonic mean normalization method. Harmonic mean always decreases with increasing number of topics. This cant be right.

```{r}
K       <- 20
G       <- 5000
alpha   <- 0.02
eta     <- 0.02
ntopics <- seq(2, 20, by=1) 

## Better something like this. 
## These are about 225 models, which runs about 4 minutes respectively on 4 cores, which will result in about 15 hrs compute time.
# params <- expand.grid(K=20, G=5000,
#                       alpha=seq(0.01, 0.1, length.out = 5),
#                       eta=seq(0.01, 0.1, length.out = 5),
#                       ntopics=seq(2, 10, by=1))
```

# Model Tuning
With the parameters defined above we can now go on and train our set of models. For parallel computing we will define our worker function, which will bind all variables needed during training to the global environment, such that they are available for each core.
TODO: Proper explaination on http://www.win-vector.com/blog/2016/01/parallel-computing-in-r/

# Multi CPU Computation
```{r, eval=FALSE}
set.seed(357)

worker <- function() {
  bindToEnv(objNames=c("documents", "vocab", "G", "alpha", "eta"))
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
t2 - t1

save(models, file="ldaModels_NIPS2015.rda")
```

# Single CPU Computation
```{r, eval=FALSE}
t1 <- Sys.time()
models = lapply(ntopics, lda::lda.collapsed.gibbs.sampler(documents = documents, K = k, vocab = vocab, 
                                                          num.iterations = G, alpha = alpha,
                                                          eta = eta, initial = NULL, burnin = 0,
                                                          compute.log.likelihood = TRUE))
t2 <- Sys.time()
t2 - t1
```

# Model Selection
We will select our model based on harmonic mean maximization.
TODO: Check http://epub.wu.ac.at/3558/1/main.pdf for proper explanation and comparison to other performance values.
TODO: Fix harmonic mean calculation.
```{r}
harmonicMean = function(logLikelihoods, precision=2000L) {
  llMed = median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

logLiks = lapply(models, function(L)  L$log.likelihoods)
harmMeans = sapply(logLiks, function(h) harmonicMean(h))
k = which(harmMeans==max(harmMeans))

plot(ntopics, harmMeans, type = "l")

k=8 # As long as harmMean maximization does not work use 8-topic model, as it seems to be reasonable based t-SNE projection
model = models[[k]]
```

# Get the top 5 words defining the first 5 topics
```{r}
N <- 5 
top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
top.words.df <- as.data.frame(top.words)
colnames(top.words.df) <- ntopics[1]:ntopics[k]

knitr::kable(top.words.df[,1:5], caption = "Top 5 terms per topic")
```

# Get the top 5 documents assigned to the first 5 topics
```{r}
top.documents <- top.topic.documents(model$document_sums, 
                                     num.documents = 20, 
                                     alpha = alpha)
top.documents.df <- as.data.frame(top.documents)
colnames(top.documents.df) <- ntopics[1]:k

top.documents.df.part <- head(top.documents.df, 10)

topic_titles <- data.frame(Topic1=papers[as.numeric(top.documents.df.part[ ,1]),]$Title,
                           Topic2=papers[as.numeric(top.documents.df.part[ ,2]),]$Title,
                           Topic3=papers[as.numeric(top.documents.df.part[ ,3]),]$Title,
                           Topic4=papers[as.numeric(top.documents.df.part[ ,4]),]$Title,
                           Topic5=papers[as.numeric(top.documents.df.part[ ,5]),]$Title)

knitr::kable(topic_titles, caption = "Top 10 titles per topic")
```

# Get topics with maximum log likelihood for docs
First we will compute to which proportion a document belongs to a topic. As zero values and NAs will be a problem in the succeeding steps we will add a small number to each element in the topic proportion matrix. The topic with the maximum proportion value will then be assigned to the document.
TODO: Compare this assignment with the top.documents matrix. The result should be the same, and I don't belive it is.
TODO: Plot an example proportion vector for a few documents.
```{r}
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
```

# Compute similarity between documents
A document is defined as a mixture of topics, each with a certain probability. In other words a document is to some proportion, part of each topic. Given two proportion vectors, a similarity can be computed between two documents. A similarity measure between two probability distributions, can be computed using the Jensen-Shannon divergence (JSD) (TODO: citation), which can be derived from the Kullback-Leibler divergence. (TODO: LaTeX formula of JSD) The JSD is defined as follows

```{r}
## Compute Jensen-Shannon Divergence between documents
## p,q probability distribution vectors R^n /in [0,1]
JSD <- function(p, q) {
  m <- 0.5 * (p + q)
  divergence <- 0.5 * (sum(p * log(p / m)) + sum(q * log(q / m)))
  return(divergence)
}
```

With this we can compute the pairwise similarity between each document.
```{r}
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
```

```{r}
X_dist <- sqrt(X) # compute Jensen-Shannon Distance
```

## Clustering and Dimension Reduction of Jensen-Shannon Distance matrix
To visualize the results we have to reduce the dimensionality of our document similarity matrix. To do this we need a distance matrix. Taking the square root of the JSD matrix results in a metric called Jensen-Shannon distance, which can be used in hirarchical clustering as well as, dimensionality reduction algorithms.
```{r, eval=FALSE, echo=FALSE}
library(apcluster)
## run affinity propagation
apres <- apcluster(X_dist, details=TRUE)
show(apres)

## plot information about clustering run
plot(apres)

## plot clustering result
plot(apres, X_dist)

## employ agglomerative clustering to join clusters
aggres <- aggExCluster(sim, apres)

## show information
show(aggres)
show(cutree(aggres, 2))

## plot dendrogram
plot(aggres)

## plot clustering result for k=2 clusters
plot(aggres, X_dist, k=2)

## plot heatmap
heatmap(apres, sim)
```

For exploratory purposes we will embedd the distance matrix onto a 2-dimensional plane using different projection methods, Multidimensional Scaling (TODO: citation), Principal Component Analysis (TODO: cite) and t-SNE (TODO: cite).

TODO: init_dims and perplexity has to be tuned for t-SNE. It is yet unclear how to do this properly
```{r}
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
```

Now let's have a look at the results using the interactive plotting library rbokeh, with which it is possible to select certain clusters in each projection, a method called linked brushing.

```{r}
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
```

# LDAVis Visualization
```{r}
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
```

# References

http://winvector.github.io/Parallel/PExample.html
http://www.win-vector.com/blog/2016/01/parallel-computing-in-r/
https://eight2late.wordpress.com/2015/09/29/a-gentle-introduction-to-topic-modeling-using-r/
https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf
https://www.aaai.org/ocs/index.php/ICWSM/ICWSM12/paper/viewFile/4645/5021
http://epub.wu.ac.at/3558/1/main.pdf

# Session Info
```{r}
sessionInfo()
```
