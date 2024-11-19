# ODA: learning solution operator for the variational data assimilation problem

[view overleaf](https://www.overleaf.com/read/ktxhjhxfhkcf#c15a2d)


## Setup
1. clone this repository `git clone https://github.com/jaeminoh/DALS.git`
2. enter to the directory: `cd oda`
3. install via pip: `pip install -e .`
4. install dependencies: `pip install -r requirements.txt`

## Examples
1. `cd Lorenz96` or `cd Kursiv`
1. To generate training and test data, run `python make_data.py` in each directory.
2. Run `sh main.sh`

### Lorenz 96 (Lorenz96)

Forward Euler:


https://github.com/user-attachments/assets/185b85fc-f90a-4674-8a12-97b1e0d83dab


Noisy Observation:


https://github.com/user-attachments/assets/15147f46-a995-46ec-af6b-b12b125c6eec


Combination (Assimilation):


https://github.com/user-attachments/assets/c94dee6f-509d-48c4-90a8-0b5eb80d7afd



### Kuramoto-Sivashinsky (Kursiv)

Forward Euler:


https://github.com/user-attachments/assets/4085b54c-39aa-479f-8083-8cdac189afca


Noisy Observation:


https://github.com/user-attachments/assets/f2948dc0-7fbb-434d-bb4a-e188f6dba823


Combination (Assimilation):


https://github.com/user-attachments/assets/b9ff6202-f799-47de-b40c-89899b23bb08







