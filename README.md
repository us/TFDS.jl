# TFDS.jl
## Usage
Clone Repo
```
git clone https://github.com/us/TFDS.jl.git
```
Then import package
```
using Pkg; Pkg.activate("TFDS.jl"); using TFDS
```
Load Mnist Dataset
```
test, train = tfds.as_numpy(load("mnist", batch_size=-1))
test_X, test_Y = test.second["image"], test.second["label"]
train_X, train_Y = train.second["image"], test.second["label"]
```
```
julia> typeof(train_X)
Array{UInt8,4}

julia> typeof(train_Y)
Array{Int64,1}
```
