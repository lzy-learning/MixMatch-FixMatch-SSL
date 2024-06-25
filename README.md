# MixMatch-FixMatch-SSL
SYSU Pattern Recognition Third Assignment for the Second Half of 2023



### Run

```shell
> python.exe .\main.py -h

usage: main.py [-h] [--algorithm {mix,fix}] [--labeled_num {40,250,4000}]
               [--train {True,False}]

Parse command line arguments for the project main function

options:
  -h, --help            		show this help message and exit
  --algorithm {mix,fix} 		Specify the algorithm to use (MixMatch or FixMatch). Default is mixmatch.
  --labeled_num {40,250,4000}   Specify the amount of labeled data, choose from: 40, 250, 4000. Default is 250.
  --train {True,False}  		True for training the model. Default is True.
```

#### Example

```shell
python main.py --algorithm=mix --labeled_num=40 --train=True
# ensure you had trained the model and store the checkpoint file
python main.py --algorithm=mix --labeled_num=40 --train=False		
```

