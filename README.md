# Resordan

REsident Space Object RAdar Data ANalysis


# Install

## Pre requirements
    - python
    - Java/8
    - Maven
    - JCC
    - astropy
    - spacetrack
    - similaritymeasures
    - sorts

## Step 1 - Create virtual environment
```sh
python -m venv ~/venv/resordan
source ~/venv/resordan/bin/activate
pip install --upgrade pip
```

## Step 2 - Install Resordan

Install resordan package within virtual environment

```sh
cd ~/git
git clone https://github.com/danielk333/resordan
cd ~/git/resordan
pip install -e .[develop]
```

## Step 3 - Install SORTS

The `correlator` module depends on SORTS [https://github.com/danielk333/SORTS](https://github.com/danielk333/SORTS).

Follow install instructions for SORTS. Install within the virtual environment. 

```sh
cd ~/git
git clone --branch develop https://github.com/danielk333/SORTS
cd ~/git/SORTS
pip install -e .[develop,mpi,plotting] –verbose
```

