# Resordan

REsident Space Object RAdar Data ANalysis

# Install

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
Additionally run script to install Orekit

```sh
export JDK_LOC=$JAVA_HOME
export JCC_JDK=$JAVA_HOME
./install_orekit.sh jcc
./install_orekit.sh check
./install_orekit.sh build
./install_orekit.sh install
```

