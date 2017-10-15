abstract type AbstractCallback end

mutable struct Bucket
  iter::Int
  names::Vector{String}
  cur_mean::Dict{String, Float64}
  cur_error::Dict{String, Float64}
  stop_condition::Bool
  end_iteration::Int
  num_parallel_tree::Int
  params::Vector{Tuple{String, String}}

  # Early Stop Results
  best_iteration::Int
  best_score::Float64
  best_score_error::Float64
  best_ntree_limit::Int

  bst

  function Bucket()
    new(1, [], Dict(), Dict(), false, 1, 1, [], 1, 0, 0, 1, nothing)
  end
end

finalize!(cb::AbstractCallback, bck::Bucket) = nothing
pre_iter!(cb::AbstractCallback, bck::Bucket) = nothing
post_iter!(cb::AbstractCallback, bck::Bucket) = nothing

function finalize!{T <: AbstractCallback}(cbs::Vector{T}, bck::Bucket)
  for cb in cbs
    finalize!(cb, bck)
  end
end

function pre_iter!{T <: AbstractCallback}(cbs::Vector{T}, bck::Bucket)
  for cb in cbs
    pre_iter!(cb, bck)
  end
end

function post_iter!{T <: AbstractCallback}(cbs::Vector{T}, bck::Bucket)
  for cb in cbs
    post_iter!(cb, bck)
  end
end


mutable struct EarlyStopCallback <: AbstractCallback
  initialized::Bool
  verbose::Bool
  best_iteration::Int
  best_score::Float64
  best_msg::Vector
  best_ntree_limit::Int
  maximize::Bool
  metric_name::String
  stopping_rounds::Int

  function EarlyStopCallback(stopping_rounds; maximize::Bool = false, metric_name::String = "", verbose::Bool = true)
    new(false, verbose, 1, Inf, [], 1, maximize, metric_name, stopping_rounds)
  end
end

function init!(cb::EarlyStopCallback, bck::Bucket)
  if cb.metric_name == ""
    cb.metric_name = bck.names[end]
  end

  if cb.maximize
    cb.best_score = -Inf
  else
    cb.best_score = Inf
  end

  cb.initialized = true
end

function post_iter!(cb::EarlyStopCallback, bck::Bucket)
  if !(cb.initialized)
    init!(cb, bck)
  end
  i = bck.iter
  bck.end_iteration = i
  score = bck.cur_mean[cb.metric_name]

  if (cb.maximize && score > cb.best_score) || (!cb.maximize && score < cb.best_score)
    cb.best_score = score
    cb.best_msg = [i, bck.cur_mean, bck.cur_error]
    cb.best_iteration = i
    cb.best_ntree_limit = i*bck.num_parallel_tree
  else
    if (i - cb.best_iteration >= cb.stopping_rounds)
      bck.stop_condition = true
    end
  end

  if cb.verbose
    println("Best iteration: ", cb.best_iteration)
    println("Best value: ", cb.best_msg)
  end
end

function finalize!(cb::EarlyStopCallback, bck::Bucket)
  bck.best_iteration = cb.best_iteration
  bck.best_score = cb.best_score
  bck.best_score_error = cb.best_msg[3][cb.metric_name]
  bck.best_ntree_limit = cb.best_ntree_limit
end

####################################

"""
Callback for printing the result of evaluation
"""
mutable struct EvaluationPrint <: AbstractCallback
  period::Int
  showsd::Bool
  last_iter::Int

  function EvaluationPrint(; period::Int = 1, showsd::Bool = true)
    new(period, showsd, 0)
  end
end

function post_iter!(cb::EvaluationPrint, bck::Bucket)
  if (bck.iter % cb.period == 0) || (bck.iter == 1)
    cb.last_iter = bck.iter
    msg = @sprintf("[%s] ", bck.iter)
    for name in bck.names
      msg *= @sprintf("%s %s ", name, bck.cur_mean[name])
      if cb.showsd
        msg *= @sprintf("%s-sd %s ", name, bck.cur_error[name])
      end
    end
    println(msg)
  end
end

function finalize!(cb::EvaluationPrint, bck::Bucket)
  if cb.last_iter != bck.iter
    msg = @sprintf("[%s] ", bck.iter)
    for name in bck.names
      msg *= @sprintf("%s %s ", name, bck.cur_mean[name])
      if cb.showsd
        msg *= @sprintf("%s-sd %s ", name, bck.cur_error[name])
      end
    end
    println(msg)
  end
end

####################################

"""
Callback for logging the evaluation history
"""
mutable struct EvaluationLog <: AbstractCallback
  f::IOStream
  separator::Union{String, Char}
  header::Bool
  showsd::Bool
  additional_info::Vector
  finalize::Bool
  name::String

  function EvaluationLog(name::String = "xgboost"; random_token::Bool = true, timed::Bool = true, token_length::Int = 4, separator::Union{String, Char} = '\t', header = true, showsd::Bool = true, additional_info = [], finalize::Bool = true, output_dir = ".")
    if timed
      name = Dates.format(now(), "yyyymmddTHHMMSS-") * name
    end
    if random_token
      name = name * "-" * join(rand(vcat('a':'z', 'A':'Z', '0':'9'), token_length))
    end
    name = joinpath(output_dir, name)
    res_name = name
    name = name * ".csv"

    if !ispath(output_dir)
      mkpath(output_dir)
    end
    f = open(name, "w")
    new(f, separator, header, showsd, additional_info, finalize, res_name)
  end
end

get_filename(cb::EvaluationLog) = basename(cb.name)

function post_iter!(cb::EvaluationLog, bck::Bucket)
  if cb.header
    msg = String["id"]
    for name in bck.names
      push!(msg, name)
      if cb.showsd
        push!(msg, @sprintf("%s-sd", name))
      end
    end
    msg = vcat(msg, [string(itm[1]) for itm in cb.additional_info])

    write(cb.f, join(msg, cb.separator))
    write(cb.f, "\n")
    cb.header = false
  end

  msg = String[@sprintf("%s", bck.iter)]
  for name in bck.names
    push!(msg, @sprintf("%s", bck.cur_mean[name]))
    if cb.showsd
      push!(msg, @sprintf("%s", bck.cur_error[name]))
    end
  end
  msg = vcat(msg, [string(itm[2]) for itm in cb.additional_info])
  write(cb.f, join(msg, cb.separator))
  write(cb.f, "\n")
  flush(cb.f)
end

function finalize!(cb::EvaluationLog, bck::Bucket)
  if cb.finalize
    close(cb.f)
  end
  open(cb.name*".meta", "w") do f
    for itm in bck.params
      write(f, @sprintf("%s%s%s\n", itm[1], cb.separator, itm[2]))
    end
  end
end

function finalize!(cb::EvaluationLog)
  close(cb.f)
end

####################################
"""
Callback for saving a model file.
"""

struct SaveModel <: AbstractCallback
  save_period::Union{Int, Void}
  save_name::String
  additional_info::String

  function SaveModel(; output_dir = ".", save_period::Union{Int, Void} = nothing, save_name::String = "xgboost_model", additional_info = "")
    if !ispath(output_dir)
      mkpath(output_dir)
    end
    new(save_period, joinpath(output_dir, save_name), additional_info)
  end
end

function post_iter!(cb::SaveModel, bck::Bucket)
  if !isa(cb.save_period, Void) && (bck.iter % cb.save_period == 0)
    if isempty(cb.additional_info)
      save(bck.bst, @sprintf("%s-%05d.model", cb.save_name, bck.iter))
    else
      save(bck.bst, @sprintf("%s-%s-%05d.model", cb.save_name, cb.additional_info, bck.iter))
    end
  end
end

function finalize!(cb::SaveModel, bck::Bucket)
  if cb.save_period isa Void
    if isempty(cb.additional_info)
      save(bck.bst, @sprintf("%s-%05d.model", cb.save_name, bck.iter))
    else
      save(bck.bst, @sprintf("%s-%s-%05d.model", cb.save_name, cb.additional_info, bck.iter))
    end
  end
end
