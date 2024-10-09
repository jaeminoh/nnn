using NCDatasets, Dates, Statistics

"""
A function to read temperature data.
"""
function read_temperature(year::Int, return_others::Bool=false)
    @assert 1979 ≤ year ≤ 2018
    d = NCDataset("ERA5/temperature_850/temperature_850hPa_$(year)_5.625deg.nc")
    temperature = d[:t][:, :, :]
    if return_others
        time = d[:time][:]
        lon = d[:lon][:]
        lat = d[:lat][:]
        return temperature, lon, lat, time
    end
    return temperature # (lon, lat, 1, time) shaped.
end

"""
hourly data to daily data by taking mean.
"""
function hourly_to_daily(temperature, time)
    y = year(time[1])
    date = Date.(time)
    days_of_year =  date |> Set |> collect |> sort # unique days

    # filter out yyyy-02-29
    condition = try
        Date(y, 2, 29)
    catch
        1
    end
    filter!(d -> d ≠ condition, days_of_year)

    # daily mean temperature
    daily_t = zeros(Float32, size(temperature)[1:2]..., length(days_of_year)) # (64, 32, 365)
    for (i, day) in enumerate(days_of_year)
        indices = findall(date .== day)
        _t = temperature[:,:,indices]
        _t = mean(_t; dims=ndims(_t))
        daily_t[:,:,i] .= _t[:,:,end]
    end

    return daily_t, days_of_year
end
