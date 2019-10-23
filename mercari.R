library(Matrix)
library(tidyverse)
library(lightgbm)
library(quanteda)
library(stringr)
library(tictoc)
library(glue)

train_cols <- cols(
  train_id = col_integer(),
  name = col_character(),
  item_condition_id = col_integer(),
  category_name = col_character(),
  brand_name = col_character(),
  price = col_double(),
  shipping = col_integer(),
  item_description = col_character()
)

test_cols <- cols(
  test_id = col_integer(),
  name = col_character(),
  item_condition_id = col_integer(),
  category_name = col_character(),
  brand_name = col_character(),
  shipping = col_integer(),
  item_description = col_character()
)

train <- read_tsv("../input/train.tsv", col_types = train_cols)
test <- read_tsv("../input/test.tsv", col_types = test_cols)


train <- train %>%
  filter(price != 0)

train$price_log  <- log(train$price + 1)


## Handling categories

tic("Splitting categories")
temp <- as_tibble(str_split(train$category_name, "/", n = 3, simplify = TRUE))
names(temp) <- paste0("category", 1:3)
train <- bind_cols(train, temp)

temp <- as_tibble(str_split(test$category_name, "/", n = 3, simplify = TRUE))
names(temp) <- paste0("category", 1:3)
test <- bind_cols(test, temp)
toc()

# Cleaning and some new features
tic("Data preprocessing")
train <- train %>%
  mutate(item_description = if_else(is.na(item_description) |
                                    str_to_lower(item_description) == "no description yet",
                                    "nodescription",
                                    item_description),
         desc_length = if_else(item_description == "nodescription", 0L, str_length(item_description)),
         na_brand = is.na(brand_name),
         brand_name = if_else(brand_name == "Céline", "Celine", brand_name)
        )

test <- test %>%
  mutate(item_description = if_else(is.na(item_description) |
                                    str_to_lower(item_description) == "no description yet",
                                    "nodescription",
                                    item_description),
         desc_length = if_else(item_description == "nodescription", 0L, str_length(item_description)),
         na_brand = is.na(brand_name),
         brand_name = if_else(brand_name == "Céline", "Celine", brand_name)
        )


# expensive_brands <- train %>%
#   group_by(brand_name) %>%
#   summarize(n= n(), median = median(price), mean = mean(price)) %>%
#   filter(n >= 10 & median >= 100)

## Handling missing values

train$category1[is.na(train$category1)] <- "missing"
train$category2[is.na(train$category2)] <- "missing"
train$category3[is.na(train$category3)] <- "missing"

test$category1[is.na(test$category1)] <- "missing"
test$category2[is.na(test$category2)] <- "missing"
test$category3[is.na(test$category3)] <- "missing"


train$brand_name[is.na(train$brand_name)] <- "missing"
test$brand_name[is.na(test$brand_name)] <- "missing"


train$category_name[is.na(train$category_name)] <- "missing"
test$category_name[is.na(test$category_name)] <- "missing"

toc()

names(train)[1] <- names(test)[1] <- "item_id"

log_prices  <- train$price_log
train$price_log <- NULL


all <- bind_rows(train, test)

tic("Descriptions dtm")
descriptions <- corpus(char_tolower(all$item_description))

description_tokens <- tokens(
  tokens_remove(tokens(descriptions,
                       remove_numbers = FALSE,
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_separators = TRUE),
                stopwords("english")),
  ngrams = 1:2
)


description_dtm <- dfm(
  description_tokens
)
toc()

tic("Descriptions tf-idf")
description_dtm_trimmed <- dfm_trim(description_dtm, min_count = 600)
description_tf_matrix <- tfidf(description_dtm_trimmed)
toc()

tic("Names dtm")
names <- corpus(char_tolower(all$name))

names_tokens <- tokens(
  tokens_remove(tokens(names,
                       remove_numbers = TRUE,
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_separators = TRUE),
                stopwords("english")),
  ngrams = 1

)


names_dtm <- dfm(
  names_tokens
)
toc()

trimmed_names_dfm <- dfm_trim(names_dtm, min_count = 30)

tic("Preparing data for modelling")
sparse_matrix <- sparse.model.matrix(
  ~item_condition_id +
    shipping +
    na_brand +
    category1 +
    category2 +
    category3 +
    desc_length +
    brand_name,
  data = all)

## Fix for cbind dfm and sparse matrix
class(description_tf_matrix) <- class(sparse_matrix)
class(trimmed_names_dfm) <- class(sparse_matrix)

aaa <- cbind(
  sparse_matrix, # basic features
  description_tf_matrix,  # description
  trimmed_names_dfm # name
)

rownames(aaa) <- NULL

glue("Number of features: {dim(aaa)[2]}")

sparse_train <- aaa[seq_len(nrow(train)), ]
sparse_test  <- aaa[seq(from = (nrow(train) + 1), to = nrow(aaa)), ]

dtrain <- lgb.Dataset(sparse_train, label=log_prices)
toc()

nrounds <- 8000
param <- list(
              objective = "regression",
              metric = "rmse"
             )

set.seed(333)
tic("Modelling")

model <- lgb.train(
  params = param,
  data = dtrain,
  nrounds = nrounds,
  learning_rate = 1,
  subsample = 0.7,
  max_depth = 4,
  eval_freq = 50,
  verbose = -1,
  nthread = 4
)
toc()

tic("Predicting results")
log_predicted <- predict(model, sparse_test)
predicted <- exp(log_predicted) - 1

results <- data.frame(
  test_id = as.integer(seq_len(nrow(test)) - 1),
  price = predicted
)
toc()

write_csv(results, "submission.csv")
