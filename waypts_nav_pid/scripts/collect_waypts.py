#!/bin/bash
x=1
while [ $x -le 1000 ]
do 	
	echo "$x"
	rostopic echo -n 1 /aft_mapped_to_init | shyaml get-value pose.pose >> "track1/pose_$x.yaml"
	x=$(( $x + 1 ))
done

