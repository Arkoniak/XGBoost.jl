module CrossValidation
using XGBoost
using Base.Test

##########################################################
# Test implementation
##########################################################

function test_crossval()
  dtrain = DMatrix("../../data/agaricus.txt.train")
  dtest = DMatrix("../../data/agaricus.txt.test")
  watchlist = [(dtest, "eval"), (dtrain, "train")]

  bck = nfold_cv(dtrain, 500, 3, eta = 1, max_depth = 2, objective = "binary:logistic", silent = 1,
                 seed = 12345, callbacks = [EarlyStopCallback(10)])
end

function gini(actual, pred)
  @assert length(actual) == length(pred)
  
  total_losses = sum(actual)
  p = sortperm(pred, rev = true)
  gini_sum = sum(cumsum(actual[p]))/total_losses
  gini_sum -= (length(actual) + 1)/2

  return gini_sum/length(actual)
end

gini_normalized(actual, pred) = gini(actual, pred)/gini(actual, actual)

ginic(preds::Vector{Float32}, dtrain::DMatrix) = gini(get_label(dtrain), preds)
ginic_normalized(preds::Vector{Float32}, dtrain::DMatrix) = ("gini-error", gini_normalized(get_label(dtrain), preds))

function test_train()
  dtrain = DMatrix("../../data/agaricus.txt.train")
  dtest = DMatrix("../../data/agaricus.txt.test")
  watchlist = [("eval", dtrain), ("train", dtest)]

  xgb_params = [
    "booster" => "gbtree",
    "objective" => "binary:logistic",
    "eta" => 0.07,
    "gamma" => 0,
    "max_depth" => 5,
    "subsample" => 0.95,
    "min_child_weight" => 20
  ]
  bst = train(dtrain, 20,
              params = xgb_params,
              silent = 0,
              seed = 12345,
              feval = ginic_normalized)
end

################################################################################
# Run tests
################################################################################
@testset "CrossValidation Test" begin
  # test_crossval()
  test_train()
end

end
