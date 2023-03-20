#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Download Code
# load requied libraries --------------------------------------------------
library(data.table)


# load and transform people data ------------------------------------------
ppl <- fread("../input/people.csv")

### Recode logic to numeric
p_logi <- names(ppl)[which(sapply(ppl, is.logical))]

for (col in p_logi) {
  set(ppl, j = col, value = as.numeric(ppl[[col]]))
}
rm(p_logi)

### transform date
ppl[,date := as.Date(as.character(date), format = "%Y-%m-%d")]

# load activities ---------------------------------------------------------

# read and combine
activs <- fread("../input/act_train.csv")
TestActivs <- fread("../input/act_test.csv")
TestActivs$outcome <- NA
activs <- rbind(activs,TestActivs)
rm(TestActivs)

# Extract only required variables
activs <- activs[, c("people_id","outcome","activity_id","date"), with = F]

# Merge people data into actvitities
d1 <- merge(activs, ppl, by = "people_id", all.x = T)

# Remember, remember the 5th of November and which is test
testset <- which(ppl$people_id %in% d1$people_id[is.na(d1$outcome)])
d1[, activdate := as.Date(as.character(date.x), format = "%Y-%m-%d")]

rm(activs)

# prepare grid for prediction ---------------------------------------------

# Create all group_1/day grid
minactivdate <- min(d1$activdate)
maxactivdate <- max(d1$activdate)
alldays <- seq(minactivdate, maxactivdate, "day")
allCompaniesAndDays <- data.table(
  expand.grid(unique(
    d1$group_1[!d1$people_id %in% ppl$people_id[testset]]), alldays
  )
)


## Nicer names
colnames(allCompaniesAndDays) <- c("group_1","date.p")

## sort it
setkey(allCompaniesAndDays,"group_1","date.p")

## What are values on days where we have data?
meanbycomdate <- d1[
  !d1$people_id %in% ppl$people_id[testset],
  mean(outcome),
  by = c("group_1","activdate")
  ]

## Add them to full data grid
allCompaniesAndDays <- merge(
  allCompaniesAndDays,
  meanbycomdate,
  by.x = c("group_1","date.p"), by.y = c("group_1","activdate"),
  all.x = T
)


# design function to interpolate unknown values ---------------------------

interpolateFun <- function(x){
  
  # Find all non-NA indexes, combine them with outside borders
  borders <- c(1, which(!is.na(x)), length(x) + 1)
  # establish forward and backward - looking indexes
  forward_border <- borders[2:length(borders)]
  backward_border <- borders[1:(length(borders) - 1)]
  
  # prepare vectors for filling
  forward_border_x <- x[forward_border]
  forward_border_x[length(forward_border_x)] <- abs(
    forward_border_x[length(forward_border_x) - 1] - 0.1
  ) 
  backward_border_x <- x[backward_border]
  backward_border_x[1] <- abs(forward_border_x[1] - 0.1)
  
  # generate fill vectors
  forward_x_fill <- rep(forward_border_x, forward_border - backward_border)
  backward_x_fill <- rep(backward_border_x, forward_border - backward_border)
  forward_x_fill_2 <- rep(forward_border, forward_border - backward_border) - 
    1:length(forward_x_fill)
  backward_x_fill_2 <- 1:length(forward_x_fill) -
    rep(backward_border, forward_border - backward_border)
  
  #linear interpolation
  vec <- (forward_x_fill + backward_x_fill)/2
  
  x[is.na(x)] <- vec[is.na(x)]
  return(x)
}

