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

papers = fread("~/Datasets/kaggle/NIPS2015_papers/Papers.csv")
docs = papers$Abstract

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

get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

D <- length(documents)  # number of documents
W <- length(vocab)  # number of terms in the vocab
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document
N <- sum(doc.length)  # total number of tokens in the data
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus

params <- expand.grid(G=1000,
                      k=seq(2, 10, by=1),
                      alpha=seq(0.01, 100, length.out=10),
                      eta=seq(0.01, 100, length.out=10))
params <- setNames(split(params, seq(nrow(params))), rownames(params))

set.seed(357)

worker <- function() {
  bindToEnv(objNames=c("documents", "vocab"))
  function(params) {
    k <- params$k
    G <- params$G
    alpha <- params$alpha
    eta <- params$eta
    lda::lda.collapsed.gibbs.sampler(documents = documents, K = k, vocab = vocab, 
                                     num.iterations = G, alpha = alpha, 
                                     eta = eta, initial = NULL, burnin = 0,
                                     compute.log.likelihood = TRUE)
  }
}

t1 <- Sys.time()
cluster <- startCluster()
models <- parLapply(cluster, params, worker())
shutDownCluster(cluster)
t2 <- Sys.time()
t2 - t1

save(models, file="ldaModels_NIPS2015.rda")

logL_prior <- model$log.likelihoods[1, ] # log likelihood (including the prior) per iteration
logL_obs <- model$log.likelihoods[2, ] # log likelihood of the observations conditioned on the assignments per iteration

harmonicMean = function(logLikelihoods, precision=2000L) {
  llMed = median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

logLiks = lapply(models, function(L)  L$log.likelihoods[1,])
logLiks.df <- as.data.frame(logLiks)
logLiks.df$Iteration <- rownames(logLiks.df)
colnames(logLiks.df) <- 1:ncol(logLiks.df)
molten_logLiks <- melt(logLiks.df)
colnames(molten_logLiks) <- c("Iteration", "NumberOfTopics", "logLikelihood")

ggplot(data=molten_logLiks) + 
  geom_line(aes(x=as.numeric(Iteration), y=logLikelihood, 
                color=as.factor(NumberOfTopics)))


harmMeans = sapply(logLiks, function(h) harmonicMean(h))
maxLiks <- sapply(logLiks, function(h) max(h))
k = which(harmMeans==max(harmMeans))

plot(1:length(maxLiks), maxLiks, type = "l")

k=8 # As long as harmMean maximization does not work use 8-topic model, as it seems to be reasonable based t-SNE projection
model = models[[k]]

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
