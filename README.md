Algorithm based on Chao-Chen Wu(chaochen.wu@gmail.com) for matlab, i just translate it for C and OpenCl

How to use!

Compile:

Make

execute:

./Hyper [route_file] [Num_endmembers] 


remember we need .hrd and .bsq files to make it work

ie: ./Hyper /Cuprite/Cuprite 19

./Hyper /Hydice/Hydice 19


//--------------Coming soon------------------------------//

Open_cl:

cd /Opencl

make

./Hyper [route_file] [num_endmembers] [c|g]   

//c for CPU and g for GPU



