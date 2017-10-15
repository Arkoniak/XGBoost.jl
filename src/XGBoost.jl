__precompile__()

module XGBoost

include("xgboost_lib.jl")
using .CrossValidation

export DMatrix, Booster
export xgboost, train, predict, save, nfold_cv, slice, get_info, set_info, get_label, dump_model, importance
export Kfold, StratifiedKfold
export EarlyStopCallback, EvaluationPrint, EvaluationLog, SaveModel

end # module XGBoost
