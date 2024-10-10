module DALS

include("problems.jl")
export lorenz96!
export kursiv

include("data_routines.jl")
export read_normalized_daily_temperature

end

#include("surrogate.jl")
#export init
