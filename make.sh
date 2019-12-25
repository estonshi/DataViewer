#!/bin/bash

function check_answer(){
	flag=0
	while [ $flag -eq 0 ]
	do
		read answer
		if [ x$answer = x"n" ]
		then
			flag=1
			return 0
		elif [ x$answer = x"y" ]
		then
			flag=1
			return 1
		else
			echo "??? Please give 'y' or 'n'. Use 'Ctrl-C' to exit."
		fi
	done
}


if [ -z $1 ]
then
	echo "Creating new conda environment named 'viewer', continue? [y/n]";
	check_answer
	if [ $? -eq 0 ]; then
		echo "Use './make.sh <env name>' to specify environment name."
		exit 1
	fi
	conda env create -f environment.yaml -n viewer
fi
	
