#!/usr/bin/env python
# coding: utf-8



# Load packages and data
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

data <- fromJSON("../input/train.json")




# unlist every variable except `photos` and `features` and convert to tibble
vars <- setdiff(names(data), c("photos", "features"))
data <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)




head(data, n = 1)

