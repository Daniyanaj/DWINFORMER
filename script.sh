#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		t) task=$OPTARG;;
        r) train="true";;
        p) predict="true";;
        
    esac
done
echo $name	


if ${train}
then
	
	cd /share/users/Daniya/nn2/dwinformer/
	CUDA_VISIBLE_DEVICES=${cuda} dwinformer_train 
fi

if ${predict}
then


	cd /share/users/Daniya/nn2/dwinformer/DATASET/dwinformer_raw/dwinformer_raw_data/Task002_Synapse/
	CUDA_VISIBLE_DEVICES=${cuda} dwinformer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr dwinformerTrainerV2_${name}
	python inference_acdc.py ${name}
fi

