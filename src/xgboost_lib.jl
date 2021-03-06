include("xgboost_wrapper_h.jl")
include("callbacks.jl")
include("crossval.jl")

# TODO: Use reference instead of array for length

type DMatrix
  handle::Ptr{Void}
  _set_info::Function

  function _setinfo{T<:Number}(ptr::Ptr{Void}, name::String, array::Vector{T})
    if name == "label" || name == "weight" || name == "base_margin"
      XGDMatrixSetFloatInfo(ptr, name,
                            convert(Vector{Float32}, array),
                            convert(UInt64, size(array)[1]))
    elseif name == "group"
      XGDMatrixSetGroup(ptr,
                        convert(Vector{UInt32}, array),
                        convert(UInt64, size(array)[1]))
    else
      error("unknown information name")
    end
  end

  function DMatrix(handle::Ptr{Void})
    dmat = new(handle, _setinfo)
    finalizer(dmat, JLFree)
    return dmat
  end

  function DMatrix(fname::String; silent = false)
    handle = XGDMatrixCreateFromFile(fname, convert(Int32, silent))
    dmat = new(handle, _setinfo)
    finalizer(dmat, JLFree)
    return dmat
  end

  function DMatrix{K<:Real, V<:Integer}(data::SparseMatrixCSC{K,V}, transposed::Bool = false;
                                        kwargs...)
    handle = (transposed ? XGDMatrixCreateFromCSCT(data) : XGDMatrixCreateFromCSC(data))
    for itm in kwargs
      _setinfo(handle, string(itm[1]), itm[2])
    end
    dmat = new(handle, _setinfo)
    finalizer(dmat, JLFree)
    return dmat
  end

  function DMatrix{T<:Real}(data::Matrix{T}, transposed::Bool = false, missing = NaN32;
                            kwargs...)
    handle = nothing
    if !transposed
      handle = XGDMatrixCreateFromMat(convert(Matrix{Float32}, data),
                                      convert(Float32, missing))
    else
      handle = XGDMatrixCreateFromMatT(convert(Matrix{Float32}, data),
                                       convert(Float32, missing))
    end

    for itm in kwargs
      _setinfo(handle, string(itm[1]), itm[2])
    end
    dmat = new(handle, _setinfo)
    finalizer(dmat, JLFree)
    return dmat
  end

  function JLFree(dmat::DMatrix)
    XGDMatrixFree(dmat.handle)
  end
end

function get_info(dmat::DMatrix, field::String)
  JLGetFloatInfo(dmat.handle, field)
end

get_label(dmat::DMatrix) = get_info(dmat, "label")

function set_info{T<:Real}(dmat::DMatrix, field::String, array::Vector{T})
  dmat._set_info(dmat.handle, field, array)
end

function save(dmat::DMatrix, fname::String; silent = true)
  XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, silent))
end

### slice ###
function slice{T<:Integer}(dmat::DMatrix, idxset::Vector{T})
  handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Vector{Int32}, idxset - 1),
                                 convert(UInt64, size(idxset)[1]))
  return DMatrix(handle)
end

nrow(dmat::DMatrix) = XGDMatrixNumRow(dmat.handle)

type Booster
  handle::Ptr{Void}
  bck::Bucket

  function Booster(; cachelist::Vector{DMatrix} = convert(Vector{DMatrix}, []),
                   model_file::String = "")
    handle = XGBoosterCreate([itm.handle for itm in cachelist], size(cachelist)[1])
    if model_file != ""
      XGBoosterLoadModel(handle, model_file)
    end
    bst = new(handle)
    finalizer(bst, JLFree)
    return bst
  end

  function JLFree(bst::Booster)
    XGBoosterFree(bst.handle)
  end
end

### save ###
function save(bst::Booster, fname::String)
  XGBoosterSaveModel(bst.handle, fname)
end

### dump model ###
function dump_model(bst::Booster, fname::String; fmap::String="", with_stats::Bool = false)
  data = XGBoosterDumpModel(bst.handle, fmap, convert(Int64, with_stats))
  fo = open(fname, "w")
  for i in 1:length(data)
    @printf(fo, "booster[%d]:\n", i)
    @printf(fo, "%s", unsafe_string(data[i]))
  end
  close(fo)
end

makeDMatrix(data::DMatrix, label) = data
function makeDMatrix(data::String, label)
  if label != Union{}
    warning("label will be ignored when data is a file")
  end

  return DMatrix(data)
end
function makeDMatrix(data::AbstractArray, label)
  if label == Union{}
    error("label argument must be present for training, unless you pass in a DMatrix")
  end

  return DMatrix(data, label = label)
end

type CVPack
  dtrain::DMatrix
  dtest::DMatrix
  watchlist::Vector{Tuple{String, DMatrix}}
  bst::Booster
  function CVPack(dtrain::DMatrix, dtest::DMatrix, params)
    bst = Booster(cachelist = [dtrain, dtest])
    for itm in params
      XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    end
    watchlist = [("train", dtrain), ("test", dtest)]
    new(dtrain, dtest, watchlist, bst)
  end
end

### train ###
function xgboost(data, nrounds::Integer; label = Union{}, param = [], watchlist = [], metrics = [], obj = Union{}, feval = Union{}, group = [], kwargs...)
  dtrain = makeDMatrix(data, label)
  if length(group) > 0
    set_info(dtrain, "group", group)
  end

  cache = [dtrain]
  for itm in watchlist
    push!(cache, itm[2])
  end
  bst = Booster(cachelist = cache)
  XGBoosterSetParam(bst.handle, "silent", "1")
  silent = false
  for itm in kwargs
    XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    if itm[1] == :silent
      silent = itm[2] != 0
    end
  end
  for itm in param
    XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
  end
  if size(watchlist)[1] == 0
    watchlist = [("train", dtrain)]
  end
  for itm in metrics
    XGBoosterSetParam(bst.handle, "eval_metric", string(itm))
  end
  for i = 1:nrounds
    update(bst, dtrain, 1, obj)
    if !silent
      @printf(STDERR, "%s", eval_set(bst, watchlist, i, feval = feval))
    end
  end
  return bst
end

function get(arr::Vector{Pair{String, Any}}, kw::String, def::Any)
  for i in 1:length(arr)
    if arr[i][1] == kw
      return arr[j][2]
    end
  end

  return def
end

"""
eXtreme Gradient Boosting Training

*train* is an advanced interface for training an xgboost model. The *xgboost* function is a simple wrapper for *train*

# Arguments

"""
function train(data, nrounds::Integer; 
               label = Union{}, params = Dict{String, Any}(), watchlist = [], 
               metrics = [], obj = Union{}, feval = Union{},
               callbacks::Vector = AbstractCallback[],
               early_stopping_rounds = nothing, kwargs...)
  dtrain = makeDMatrix(data, label)

  cache = vcat([dtrain], DMatrix[itm[2] for itm in watchlist])
  bst = Booster(cachelist = cache)

  num_class = max(get(params, "num_class", 1), 1)
  num_parallel_tree = max(get(params, "num_parallel_tree", 1), 1)
  niter_init = 0       # can be set from xgb_model (when it is implemented)
  is_update = get(params, "process_type", ".") == "update"
  niter_skip = ifelse(is_update, 0, niter_init)
  begin_iteration = niter_skip + 1
  end_iteration = niter_skip + nrounds

  params = vcat([("silent", "1")],
                [(k, string(v)) for (k, v) in params],
                [("eval_metric", itm) for itm in metrics],
                [(string(itm[1]), string(itm[2])) for itm in kwargs])

  silent = false
  for itm in params
    XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    if itm[1] == :silent
      silent = itm[2] != 0
    end
  end
  if isempty(watchlist)
    watchlist = [("train", dtrain)]
  end

  bck = Bucket()
  bck.num_parallel_tree = num_parallel_tree
  bck.params = params
  bck.bst = bst
  for i = begin_iteration:end_iteration
    bck.iter = i
    pre_iter!(callbacks, bck)
    update(bst, dtrain, i, obj)
    names, msg = eval_set(bst, watchlist, i, feval = feval)
    bck.names = names
    bck.cur_mean = Dict([k => mean(v) for (k, v) in msg])
    bck.cur_error = Dict([k => std(v, corrected = false) for (k, v) in msg])
    post_iter!(callbacks, bck)
    if bck.stop_condition
      break
    end
  end
  finalize!(callbacks, bck)

  bst.bck = bck
  return bst
end

### update ###
function update(bst::Booster, dtrain::DMatrix, iter::Integer, obj::Function)
  pred = predict(bst, dtrain)
  grad, hess = obj(pred, dtrain)
  @assert size(grad) == size(hess)
  XGBoosterBoostOneIter(bst.handle, dtrain.handle,
                        convert(Vector{Float32}, grad),
                        convert(Vector{Float32}, hess),
                        convert(UInt64, size(hess)[1]))
end

function update(bst::Booster, dtrain::DMatrix, iter::Integer, obj::Any)
  XGBoosterUpdateOneIter(bst.handle, convert(Int32, iter), dtrain.handle)
end

update(cv::CVPack, iter::Integer, obj) = update(cv.bst, cv.dtrain, iter, obj)

### eval_set ###
function eval_set(bst::Booster, watchlist::Vector{Tuple{String, DMatrix}}, iter::Integer; feval = nothing)
  dmats = DMatrix[]
  evnames = String[]
  for itm in watchlist
    push!(dmats, itm[2])
    push!(evnames, itm[1])
  end
  res = Dict()
  names = []
  if feval isa Void
    msg = XGBoosterEvalOneIter(bst.handle, convert(Int32, iter),
                               [mt.handle for mt in dmats],
                               evnames, convert(UInt64, size(dmats)[1]))
    for nv in split(msg, "\t")[2:end]
      msg_part = split(nv, ":")
      push!(names, msg_part[1])
      res[msg_part[1]] = [parse(Float64, msg_part[2])]
    end
  else
    #@printf(STDERR, "[%d]", iter)
    for j in 1:size(dmats)[1]
      pred = predict(bst, dmats[j])  # predict using all trees
      name, val = feval(pred, dmats[j])
      push!(names, @sprintf("%s-%s", evnames[j], name))
      res[@sprintf("%s-%s", evnames[j], name)] = [val]
    end
  end

  return names, res
end

eval_set(cv::CVPack, iter::Integer; feval = nothing) = eval_set(cv.bst, cv.watchlist, iter; feval = feval)

### predict ###
function predict(bst::Booster, data; output_margin::Bool = false, ntree_limit::Integer = 0)
  if typeof(data) != DMatrix
    data = DMatrix(data)
  end

  len = UInt64[1]
  ptr = XGBoosterPredict(bst.handle, data.handle, convert(Int32, output_margin),
                         convert(UInt32, ntree_limit), len)
  return deepcopy(unsafe_wrap(Array, ptr, len[1]))
end

# TODO Change this to call to MLStats or something like that
function mknfold(dall::DMatrix, nfold::Integer, params, seed::Union{Integer, Void},
                 evals=[], stratified=true; 
                 fpreproc = nothing, kwargs = [])
  if !isa(seed, Void)
    srand(seed)
  end

  if stratified
    folds = CrossValidation.StratifiedKfold(get_label(dall), nfold)
  else
    folds = CrossValidation.Kfold(nrow(dall), nfold)
  end

  ret = CVPack[]
  for (test, train) in folds
    dtrain = slice(dall, train)
    dtest = slice(dall, test)
    if fpreproc isa Void
      tparams = params
    else
      dtrain, dtest, tparams = fpreproc(dtrain, dtest, deepcopy(params))
    end

    push!(ret, CVPack(dtrain, dtest, tparams))
  end

  return ret
end

"""
nfold_cv

The cross validation function of xgboost

# Attributes
* `data`: takes an `DMatrix`, `AbstractArray` or `String` as the input
* `params`: the list of parameters. Commonly used ones are:
"""
function nfold_cv(data, nrounds::Integer = 10, nfold::Integer = 3; 
                  label = Union{}, params=[], metrics=[], 
                  stratified = true, obj = Union{}, feval = nothing,
                  fpreproc = nothing, show_stdv = true, 
                  seed::Union{Integer, Void} = nothing, 
                  callbacks::Vector = AbstractCallback[],
                  kwargs...)
  dtrain = makeDMatrix(data, label)
  params = vcat([itm for itm in params], 
                [("eval_metric", itm) for itm in metrics],
                [(string(itm[1]), string(itm[2])) for itm in kwargs])

  cvfolds = mknfold(dtrain, nfold, params, seed, stratified, fpreproc=fpreproc)
  bck = Bucket()
  bck.params = params
  result = Dict()
  for i in 1:nrounds
    bck.iter = i
    pre_iter!(callbacks, bck)
    msg = Dict{String, AbstractArray}()
    for f in cvfolds
      update(f, i, obj)
      names, new_msg = eval_set(f, i, feval = feval)
      merge!(vcat, msg, new_msg)
      bck.names = names
    end
    bck.cur_mean = Dict([k => mean(v) for (k, v) in msg])
    bck.cur_error = Dict([k => std(v, corrected = false) for (k, v) in msg])
    post_iter!(callbacks, bck)
    if bck.stop_condition
      break
    end
  end
  finalize!(callbacks, bck)

  result["best_iteration"] = bck.best_iteration
  result["best_score"] = bck.best_score
  result["best_score_error"] = bck.best_score_error
  result["end_iteration"] = bck.end_iteration

  return(result)
end

struct FeatureImportance
  fname::String
  gain::Float64
  cover::Float64
  freq::Float64
end

function Base.show(io::IO, f::FeatureImportance)
  @printf(io, "%s: gain = %0.04f, cover = %0.04f, freq = %0.04f", f.fname, f.gain, f.cover,
          f.freq)
end

function Base.show(io::IO, arr::Vector{FeatureImportance}; maxrows = 30)
  println(io, "$(length(arr))-element Vector{$(FeatureImportance)}:")
  println(io, "Gain      Coverage  Frequency  Feature")
  for i in 1:min(maxrows, length(arr))
    @printf(io, "%0.04f    %0.04f    %0.04f     %s\n", arr[i].gain, arr[i].cover, arr[i].freq,
            arr[i].fname)
  end
end

Base.show(io::IO, ::MIME"text/plain", arr::Vector{FeatureImportance}) = show(io, arr)

function importance(bst::Booster; fmap::String = "")
  data = XGBoosterDumpModel(bst.handle, fmap, 1)

  # get the total gains for each feature and the whole model
  gains = Dict{String,Float64}()
  covers = Dict{String,Float64}()
  freqs = Dict{String,Float64}()
  totalGain = 0.
  totalCover = 0.
  totalFreq = 0.
  lineMatch = r"^[^\w]*[0-9]+:\[([^\]]+)\] yes=([\.+e0-9]+),no=([\.+e0-9]+),[^,]*,?gain=([\.+e0-9]+),cover=([\.+e0-9]+).*"
  nameStrip = r"[<>][^<>]+$"
  for i in 1:length(data)
    for line in split(unsafe_string(data[i]), '\n')
      m = match(lineMatch, line)
      if typeof(m) != Void
        fname = replace(m.captures[1], nameStrip, "")

        gain = parse(Float64, m.captures[4])
        totalGain += gain
        gains[fname] = get(gains, fname, 0.) + gain

        cover = parse(Float64, m.captures[5])
        totalCover += cover
        covers[fname] = get(covers, fname, 0.) + cover

        totalFreq += 1
        freqs[fname] = get(freqs, fname, 0.) + 1
      end
    end
  end

  # compile these gains into list of features sorted by gain value
  res = FeatureImportance[]
  for fname in keys(gains)
    push!(res, FeatureImportance(fname,
                                 gains[fname] / totalGain,
                                 covers[fname] / totalCover,
                                 freqs[fname] / totalFreq))
  end
  sort!(res, by = x -> -x.gain)
end

function importance(bst::Booster, feature_names::Vector{String})
  res = importance(bst)

  result = FeatureImportance[]
  for old_importance in res
    actual_name = feature_names[parse(Int64, old_importance.fname[2:end]) + 1]
    push!(result, FeatureImportance(actual_name, old_importance.gain, old_importance.cover,
                                    old_importance.freq))
  end

  return result
end
