module LatentDA


greet() = print("Hello World!")
export greet

include("problems.jl")
export lorenz96!
export kursiv

end # module Examples
