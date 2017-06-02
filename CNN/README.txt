To Run:

python ./retrain.py \
--bottleneck_dir=./tf_files/bottlenecks \
--how_many_training_steps 7500 \
--model_dir=../inception \
--output_graph=./tf_files/retrained_graph.pb \
--output_labels=./tf_files/retrained_labels.txt \
--summaries_dir=./tf_files/retrain_logs \
--flip_left_right \
--random_crop=10 \
--random_scale=10 \
--random_brightness=10 \
--learning_rate=0.01 \
--image_dir <IMAGE_DIR>


Images must be in the structure of:

Images
 |-----0
 	|-----img1.jpg
 	|-----img2.jpg

 |-----1