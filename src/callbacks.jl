abstract type AbstractCallback end

mutable struct Bucket
  iter::Int
  names::Vector{String}
  cur_mean::Dict{String, Float64}
  cur_error::Dict{String, Float64}
  stop_condition::Bool
  end_iteration::Int

  # cross val
  best_iteration::Int
  best_score::Float64
  best_score_error::Float64

  function Bucket()
    new(0, [], Dict(), Dict(), false, 0, 0, 0, 0)
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
  maximize::Bool
  metric_name::String
  stopping_rounds::Int

  function EarlyStopCallback(stopping_rounds; maximize::Bool = false, metric_name::String = "", verbose::Bool = true)
    new(false, verbose, 1, Inf, [], maximize, metric_name, stopping_rounds)
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
    # cb.best_ntreelimit = i*bck.num_parallel_tree
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
end

####################################
struct EvaluationPrint <: AbstractCallback
end

function post_iter!(cb::EvaluationPrint, bck::Bucket)
end
