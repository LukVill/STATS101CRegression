### Loading necessary libraries
library(tidyverse)
library(tidymodels)
library(stacks)
library(ranger)
library(xgboost)

### Reading necessary data
trainFilepath <- paste0(getwd(),"/train.csv")
testFilepath <- paste0(getwd(),"/test.csv")

train <- read.csv(trainFilepath)
test <- read.csv(testFilepath)

## Remove name from training data
train$name <- NULL

### Creating a cross-validation training split for the stack stratified by percent_dem
# Seed for reproducibility
set.seed(101)
train_split <- vfold_cv(train, v = 5, strata = percent_dem)

### Creating necessary recipes
## Original Recipe -- Takes all the raw entries in the training data, filters with tuning parameter to remove highly correlated variables
# and then shifts all predictors by one value to remove 0 entries in data. Imputes missing data using knn.
org_rec <- recipe(percent_dem ~ ., data = train) %>% 
  step_rm(id) %>% 
  step_corr(all_numeric_predictors(), threshold = tune()) %>% 
  step_mutate_at(all_numeric_predictors(), fn = function(x){x + 1}) %>% 
  step_impute_knn(all_numeric_predictors())

# This recipe takes the log and normalizes all the predictors in original recipe. This recipe will be used in our stack for 
# the linear model regression/workflow. 
log_norm_org_rec <- org_rec %>% 
  step_log(all_numeric_predictors(), base = 10) %>% 
  step_normalize(all_numeric_predictors())

## Simple Recipe -- This recipe uses a population percentage based transformation for predictors revolving around age, ethnicity and education.
# Like in the original recipe, missing values are imputed using knn, numeric predictors dealing with income and gdp are shifted up by 1 and have the log taken of
# to transform the values since these predictors are heavily right-skewed. 
# Raw data predictors are removed, some transformed predictors are removed as indicated by being commented out, and some gdp and education predictors are removed at the end.
# This recipe will be used randomforest and boosted tree models in the stack.
simple_rec <- recipe(percent_dem ~ ., data = train) %>%
  step_mutate(pct_male = x0002e / x0001e * 100) %>%
  # step_mutate(pct_female = x0003e / x0001e * 100) %>%
  step_mutate(pct_under15 = (x0005e + x0006e + x0007e) * 100 / x0001e) %>%
  step_mutate(pct_15thru19 = x0008e * 100 / x0001e) %>%
  step_mutate(pct_20thru24 = x0009e * 100 / x0001e) %>%
  step_mutate(pct_25thru34 = x0010e * 100 / x0001e) %>%
  step_mutate(pct_35thru44 = x0011e * 100 / x0001e) %>%
  step_mutate(pct_45thru54 = x0012e * 100 / x0001e) %>%
  step_mutate(pct_55thru64 = (x0013e + x0014e) * 100 / x0001e) %>%
  step_mutate(pct_65thru84 = (x0015e + x0016e) * 100 / x0001e) %>%
  # step_mutate(pct_over85 = (x0017e) * 100 / x0001e) %>%
  step_rename(median_age = x0018e) %>%
  step_mutate(pct_u18 = x0019e * 100 / x0001e) %>%
  step_mutate(pct_o16 = x0020e * 100 / x0001e) %>%
  # step_mutate(pct_o18 = x0021e * 100 / x0001e) %>%
  # step_mutate(pct_o21 = x0022e * 100 / x0001e) %>%
  step_mutate(pct_o62 = x0023e * 100 / x0001e) %>%
  # step_mutate(pct_o65 = x0024e * 100 / x0001e) %>%
  step_mutate(pct_m_over18 = x0026e * 100 / x0001e) %>%
  # step_mutate(pct_f_over18 = x0027e * 100 / x0001e) %>%
  step_mutate(pct_m_over65 = x0030e * 100 / x0001e) %>%
  # step_mutate(pct_f_over65 = x0031e * 100 / x0001e) %>%
  step_mutate(pct_1race = x0034e * 100 / x0001e) %>%
  # step_mutate(pct_over1race = x0035e * 100 / x0001e) %>%
  step_mutate(pct_white = x0037e * 100 / x0001e) %>%
  step_mutate(pct_black = x0038e * 100 / x0001e) %>%
  # step_mutate(pct_natamer = (x0039e + x0040e + x0041e + x0042e + x0043e) * 100 / x0001e) %>%
  step_mutate(pct_asian = x0044e * 100 / x0001e) %>%
  step_mutate(pct_indian = x0045e * 100 / x0001e) %>%
  step_mutate(pct_chinese = x0046e * 100 / x0001e) %>%
  step_mutate(pct_filipino = x0047e * 100 / x0001e) %>%
  step_mutate(pct_japan = x0048e * 100 / x0001e) %>%
  step_mutate(pct_korean = x0049e * 100 / x0001e) %>%
  step_mutate(pct_viet = x0050e * 100 / x0001e) %>%
  # step_mutate(pct_asian_other = x0051e * 100 / x0001e) %>%
  # step_mutate(pct_pac_isl = x0052e * 100 / x0001e) %>%
  # step_mutate(pct_nat_haw = x0053e * 100 / x0001e) %>%
  # step_mutate(pct_chamorro = x0054e * 100 / x0001e) %>%
  step_mutate(pct_samoan = x0055e * 100 / x0001e) %>%
  # step_mutate(pct_pacisl_other = x0056e * 100 / x0001e) %>%
  step_mutate(pct_native_haw = (x0052e + x0053e + x0054e + x0055e + x0056e) * 100 / x0001e) %>%
  step_mutate(pct_otherrace = x0057e * 100 / x0001e) %>%
  # step_mutate(pct_white_black = x0059e * 100 / x0001e) %>%
  step_mutate(pct_white_native = x0060e * 100 / x0001e) %>%
  step_mutate(pct_white_asian = x0061e * 100 / x0001e) %>%
  # step_mutate(pct_black_native = x0062e * 100 / x0001e) %>%
  step_mutate(pct_whitecombo = x0064e * 100 / x0001e) %>%
  # step_mutate(pct_blackcombo = x0065e * 100 / x0001e) %>%
  # step_mutate(pct_nativecombo = x0066e * 100 / x0001e) %>%
  # step_mutate(pct_asiancombo = x0067e * 100 / x0001e) %>%
  step_mutate(pct_hawcombo = x0068e * 100 / x0001e) %>%
  # step_mutate(pct_othercombo = x0069e * 100 / x0001e) %>%
  step_mutate(pct_hispanic = x0071e * 100 / x0001e) %>%
  step_mutate(pct_mexican = x0072e * 100 / x0001e) %>%
  # step_mutate(pct_puerto = x0073e * 100 / x0001e) %>%
  step_mutate(pct_cuban = x0074e * 100 / x0001e) %>%
  # step_mutate(pct_hisp_other = x0075e * 100 / x0001e) %>%
  # step_mutate(pct_not_hisp = x0076e * 100 / x0001e) %>%
  step_mutate(pct_nhisp_owhite = x0077e * 100 / x0001e) %>%
  step_mutate(pct_nhisp_oblack = x0078e * 100 / x0001e) %>%
  step_mutate(pct_nhisp_onatamer = x0079e * 100 / x0001e) %>%
  step_mutate(pct_nhisp_oasian = x0080e * 100 / x0001e) %>%
  step_mutate(pct_nhisp_ohawi = x0081e * 100 / x0001e) %>%
  # step_mutate(pct_nhisp_other= x0082e * 100 / x0001e) %>%
  # step_mutate(pct_nhisp_twoother = x0083e * 100 / x0001e) %>%
  step_rename(housing_units = x0086e) %>%
  step_mutate(pct_18t24_nohsdegree = c01_002e * 100 / c01_001e) %>%
  step_mutate(pct_18t24_hsdegree = c01_003e * 100 / c01_001e) %>%
  step_mutate(pct_18t24_somecollege = c01_004e * 100 / c01_001e) %>%
  step_mutate(pct_18t24_bachdegree = c01_005e * 100 / c01_001e) %>%
  step_mutate(pct_o25_nohsdegree = (c01_007e + c01_008e) * 100 / c01_006e) %>%
  step_mutate(pct_o25_hsdegree = c01_009e * 100 / c01_006e) %>%
  step_mutate(pct_o25_somecollege = (c01_010e + c01_011e) * 100 / c01_006e) %>%
  step_mutate(pct_o25_bachdegree = c01_012e * 100 / c01_006e) %>%
  # step_mutate(pct_o25_graddegree = c01_013e * 100 / c01_006e) %>%
  step_mutate(pct_25t34_hsormore = c01_017e * 100 / c01_016e) %>%
  # step_mutate(pct_25t34_bachormore = c01_018e * 100 / c01_016e) %>%
  # step_mutate(pct_35t44_hsormore = c01_020e * 100 / c01_019e) %>%
  step_mutate(pct_35t44_bachormore = c01_021e * 100 / c01_019e) %>%
  step_mutate(pct_45t64_hsormore = c01_023e * 100 / c01_022e) %>%
  step_mutate(pct_45t64_bachormore = c01_024e * 100 / c01_022e) %>%
  step_mutate(pct_o65_hsormore = c01_026e * 100 / c01_025e) %>%
  step_mutate(pct_o65_bachormore = c01_027e * 100 / c01_025e) %>%
  # Removed pct_18t24_bachdegree in impute
  step_impute_knn(income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, income_per_cap_2019, income_per_cap_2020, gdp_2016, gdp_2017, gdp_2018, gdp_2019, gdp_2020, pct_18t24_nohsdegree, pct_18t24_hsdegree, pct_18t24_somecollege, pct_18t24_bachdegree) %>%
  step_mutate_at(total_votes, median_age, housing_units, income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, income_per_cap_2019, income_per_cap_2020, gdp_2016, gdp_2017, gdp_2018, gdp_2019, gdp_2020, fn = function(x){x + 1}) %>%
  step_log(total_votes, median_age, housing_units, income_per_cap_2016, income_per_cap_2017, income_per_cap_2018, income_per_cap_2019, income_per_cap_2020, gdp_2016, gdp_2017, gdp_2018, gdp_2019, gdp_2020, base = 10) %>%
  step_rm(id, contains("x00"), contains("c01"), gdp_2016, gdp_2017, gdp_2018, gdp_2019, pct_18t24_nohsdegree, pct_18t24_somecollege)


## Metrics for stack
wf_metrics <- metric_set(rmse)

### Stack Ensemble: Linear Regression (log_norm_org_recipe), Boosted Tree(simple_rec), Random Forest(simple_rec) -- Best Performer
set.seed(10)
ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

## Linear Model -- Uses a tuning grid of 5 to determine threshold to remove correlated variables
linear_model_stk <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

linear_reg_wf <- workflow() %>%
  add_model(linear_model_stk) %>%
  add_recipe(log_norm_org_rec)

linear_reg_res <- linear_reg_wf %>%
  tune_grid(
    resamples = train_split,
    metrics = wf_metrics,
    grid = 5,
    control = ctrl_grid
  )


## Random Forest -- Uses parameter of mtry = 22, the value for this parameter was determined by optimizing a tuning grid for rmse.
# The model performs cross validation to become part of the stack.
r_forest_model_stk <- rand_forest(mtry = 22) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

r_forest_model_wf <- workflow() %>%
  add_model(r_forest_model_stk) %>%
  add_recipe(simple_rec)

r_forest_reg_res <- r_forest_model_wf %>%
  fit_resamples(
    resamples = train_split,
    metrics = wf_metrics,
    control = ctrl_res
  )

##  Boosted Tree -- Parameters of learn_rate and loss_reduction were determined by optimizing a tuning grid.
# Learn_rate was significant in finding a Boosted Tree model with low mrse, but loss_reduction was less important.
# This model performs cross validation as part of its stack.
boost_tree_model_stk <- boost_tree(learn_rate = .301, loss_reduction = 140) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

boost_tree_model_wf <- workflow() %>%
  add_model(boost_tree_model_stk) %>%
  add_recipe(simple_rec)

boost_tree_reg_res <- boost_tree_model_wf %>%
  fit_resamples(
    resamples = train_split,
    metrics = wf_metrics,
    control = ctrl_res
  )

## Candidates are added to the stack
election_data_stack <- stacks() %>%
  add_candidates(linear_reg_res) %>%
  add_candidates(r_forest_reg_res) %>%
  add_candidates(boost_tree_reg_res)

## Model is blended to add weights to each model(7 in total)
election_model_stack <- election_data_stack %>%
  blend_predictions()

## Training data is fitted to the stack
election_model_stack <- election_model_stack %>%
  fit_members()

## Predictions are generated for the test data
election_model_test_preds <- 
  election_model_stack %>% predict(new_data = test)

## ID is column bound with predictions
test_preds_stack <- test %>%
  select(id) %>%
  bind_cols(election_model_test_preds) %>% 
  rename(percent_dem = .pred)

head(test_preds_stack, n=10)



