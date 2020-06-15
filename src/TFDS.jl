module TFDS

using PyCall

const tfds = PyNULL()
export tfds,
    Split,
    load,
    as_numpy,
    builder,
    disable_progress_bar,
    is_dataset_on_gcs,
    list_builders,
    ReadConfig


function __init__()
    copy!(tfds, pyimport_conda("tensorflow_datasets", "tensorflow_datasets"))
end

function list_builders()
    tfds.list_builders()
end

# function load(args...; kwargs...)
#     tfds.as_numpy(tfds.load(args..., batch_size=-1; kwargs...))
# end

macro delegate(f_list...)
    blocks = Expr(:block)
    for f in f_list
        block = quote
            function $(esc(f))(args...; kwargs...)
                tfds.$(f)(args...; kwargs...)
            end
        end
        push!(blocks.args, block)
    end
    blocks
end

@delegate load Split as_numpy builder disable_progress_bar is_dataset_on_gcs list_builders ReadConfig

end # module
