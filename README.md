# lyft
run ->
`python main.py`

to create submission

`python eval.py --mode 'sub' --ckpt ...[pth_to_ckpt]` 

to validate a model

`python eval.py --mode 'eval' --ckpt ...[pth_to_ckpt]` 

there are two ways of validating which are `sample.zarr` and `validate.zarr`. 
we can switch between them by modifying `val_mode = "validate"` in eval.py. (str: validate -> validate.zarr | str: sample -> sample.zarr)

> `validate.zarr` returns a validation score that is 1-2 points above public LB.

> `sample.zarr` small subset for fast evaluation, returns a validation score 3-4 points below public LB.


## Training times
