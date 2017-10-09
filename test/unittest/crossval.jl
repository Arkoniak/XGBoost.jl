module CrossValidation
using XGBoost
using Base.Test

##########################################################
# Test implementation
##########################################################

function test_basic()
  dtrain = DMatrix("../../data/agaricus.txt.train")
  dtest = DMatrix("../../data/agaricus.txt.test")
  watchlist = [(dtest, "eval"), (dtrain, "train")]

  bst = nfold_cv(dtrain, 500, 3, eta = 1, max_depth = 2, objective = "binary:logistic", silent = 1,
                 seed = 12345, callbacks = [EarlyStopCallback(10)])
end

################################################################################
# Run tests
################################################################################
@testset "CrossValidation Test" begin
  test_basic()
end

end
