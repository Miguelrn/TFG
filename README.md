# TFG - 
One of the main problems in the analysis of Hyperespectral data is the time it takes to process. So in order to improve this,
we are developing a series of algorithms that are a well know in the comunity, in OpenCl, so the computational time is reduced in a scale
such that can be process in Real Time.


#How to use
you just need to provide whit 2 images, a header (.hdr) and the image (.bsq) 
You can also try 'Full-Chain' that allow u process the whole unmixing process, but u also can process just a part of the chain, the natura
order is GENE -> SGA -> LSU
Just hit Make to compile and you will need some parameters:
./full-chain 'image route' 'Max number of endmembers' 'probability of error' 'local Size' '0|1|2' '1|2'
image route: the route where both header and image are stored
Max number of endmeber: an estimation of the maximun number of endmember we will find
probability of error: the error we allow finding the endmembers
local size: in order to play whit the size of the work groups
0|1|2: mean kind device you are using,  0 -> CPU, 1 -> GPU, 2 -> accelerator
1|2: so u can choose between ViennaCl (1), ClMagma(2), this is just for LSU algorithm

#GENE
This algorithm is dedicated to extract the number of endmembers
./Gene 'image route' 'Max number of endmembers' 'probability of error'

#SGA
This one provides the location of that endmembers previous calculated
./hyper 'image route' 'number of endmebers' 'localsize' '0|1|2'

#LSU
Finaly this algorithm make the final image of abundances.
./SCLSU 'image route' 'endmembers route' 'output abundances'


for this proyect has been used OpenCl, ViennaCl & ClMagma, u als will need boost ublas and cublas in order to make it work
