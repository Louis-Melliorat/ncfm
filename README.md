# ncfm
Kaggle - The Nature Conservancy Fisheries Monitoring


### Start Tensorflow with Docker : 2 solutions
1) Download the official tensorflow image + add pandas, opencv and sklearn
docker run -p 8888:8888 -p 6006:6006 -v host_folder:container_folder tensorflow/tensorflow:0.11.0
- 8888 / 6006 mapping ports of container to host Jupyter notebook and Tensorboard
- -v host_folder:container_folder enables sharing a folder between the host and the container

2) Download the following image that contains all needed packages. 
docker run -p 8888:8888 -p 6006:6006 -v host_folder:container_folder louismlr/modified-tensorflow

To run commands inside a container
- docker exec -it \<container ID> bash
