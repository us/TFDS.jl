module TFDS

using PyCall

const tfds = PyNULL()
export core,
    as_numpy,
    decode,
    download,
    features,
    units,
    GenerateMode,
    builder,
    builder_cls,
    list_builders,
    load,
    ReadConfig,
    Split,
    testing,
    disable_progress_bar,
    is_dataset_on_gcs,
    show_examples,
    visualization,
    __version__


function __init__()
    copy!(tfds, pyimport_conda("tensorflow_datasets", "tensorflow_datasets"))
end

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

@delegate core
    as_numpy
    decode
    download
    features
    units
    GenerateMode
    builder
    builder_cls
    list_builders
    load
    ReadConfig
    Split
    testing
    disable_progress_bar
    is_dataset_on_gcs
    show_examples
    visualization
    __version__

end
