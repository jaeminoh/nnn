# Data Assimilation

Every mathematical model which is designed for realistic phenomenon cannot avoid simplifying assumptions, and thus contains systematic errors.
Primitive equation is the most famous partial differential equation which is designed to represent weather as an initial value problem.
However, due to the earth's global scale and drops' small scale, there is always an error between observation and prediction based on the model.
For a long time simulation, this systematic errors accumulate, then the simulation is not reliable at all.

Data assimilation aims to correct this systematic error based on observations.
- Satelite data contains lots of missing values. DA can produce complete gridded data. (post-processing)
- More accurate initial condition. (can be viewed as a pre-processing)