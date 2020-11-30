# motion prediction models for self-driving vehicles [WIP]

run ->
`python main.py`

to create submission

`python eval.py --mode 'sub' --ckpt ...[pth_to_ckpt]` 

to validate a model

`python eval.py --mode 'eval' --ckpt ...[pth_to_ckpt]` 

## Training times
|model|frames|raster s.|pixel s.|batch s.|worker|it|total (epoch)|
|---|---|---|---|---|---|---|---|
|effnet-b1|65|300x300|0.4,0.4|16|32|2.54s/it|123h|
|effnet-b1|65|300x300|0.4,0.4|16|4|2.65s/it|128h|
|effnet-b1|65|300x300|0.4,0.4|32|8|4.90s/it|118h|
|effnet-b1|65|300x300|0.4,0.4|64|16|9.37s/it|116h|
|effnet-b7|25|224x160|0.4,0.4|64|16|4.04s/it|48h|
|effnet-b7|65|300x300|0.4,0.4|16|4|2.67s/it|130h|
