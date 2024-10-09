using DALS

y = 1979
t, _, _, time = read_temperature(y, true)
t, time = hourly_to_daily(t, time)
