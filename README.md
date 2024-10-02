# PIORF: Physics-Informed Ollivier--Ricci Flow for Long-Range Interactions in Mesh Graph Neural Networks


## Set environment
The environment can be set up using either `environment.yaml` file or manually installing the dependencies.
### Using an environment.yaml file
```
conda env create -f environment.yaml
```

## Requirements
- tensorflow-gpu==2.8.0
- dm-sonnet==2.0.1
- protobuf==3.20.0
- networkx==2.8.6
- GraphRicciCurvature==0.5.3.1
- scipy==1.12.0
- numba==0.59.0
- pandas==2.0.0

## Download datasets
We host the datasets on this [link](https://figshare.com/s/06f3782d7ee7d23d9d31)
All data gets downloaded the `data/cylinder_flow` directory.


## How to run
To run each experiment, navigate into `PIORF-main`. Then, run the following command:

### Train a model:
```
python -m run_model --mode=train --model=MGN
python -m run_model --mode=train --model=MGN --rewire=DIGL 
python -m run_model --mode=train --model=MGN --rewire=SDRF
python -m run_model --mode=train --model=MGN --rewire=FoSR
python -m run_model --mode=train --model=MGN --rewire=BORF
python -m run_model --mode=train --model=MGN --rewire=PIORF
```

### Generate trajectory rollouts:
```
python -m run_model --mode=eval --model=MGN
python -m run_model --mode=eval --model=MGN --rewire=DIGL 
python -m run_model --mode=eval --model=MGN --rewire=SDRF
python -m run_model --mode=eval --model=MGN --rewire=FoSR
python -m run_model --mode=eval --model=MGN --rewire=BORF
python -m run_model --mode=eval --model=MGN --rewire=PIORF
```
