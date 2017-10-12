__precompile__()

module XGBoost

include("xgboost_lib.jl")

export DMatrix, Booster
export xgboost, train, predict, save, nfold_cv, slice, get_info, set_info, get_label, dump_model, importance
export EarlyStopCallback

end # module XGBoost
