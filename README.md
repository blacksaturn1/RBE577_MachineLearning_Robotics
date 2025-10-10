
# RBE577_MachineLearning_Robotics

---

## Sample Project Image

![Sample Project Image](https://via.placeholder.com/600x200.png?text=Sample+Project+Image)

---

## HW1

**Implemented the algorithms and data studied in the paper:**
> "Constrained control allocation for dynamic ship positioning using deep neural network"

### Setup
1. Create a virtual environment with Python 3.10.12:
	```bash
	python3 -m venv .venv
	source ./.venv/bin/activate
	pip install -r requirements.txt
	```

### Running Training
Run:
```bash
python hw1/src/main.py
```

---

## HW2

### Setup Python Imports
Install requirements:
```bash
pip install -r hw2/requirements.txt
```

### Download Data
Change to the data directory, set permissions, and execute the script:
```bash
cd hw2/src/data
sudo chmod 777 ./download_data.sh
./download_data.sh
```
### Start Tensorboard
```bash
tensorboard --logdir runs
```