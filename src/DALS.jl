module DALS

include("problems.jl")
export lorenz96!
export kursiv

include("read_temperature.jl")
export read_temperature
export hourly_to_daily

end

#include("surrogate.jl")
#export init
