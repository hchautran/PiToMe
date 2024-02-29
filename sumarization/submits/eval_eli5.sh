
CHECKPOINTS_DIR=/path/to/saved_checkpoints
CUDA_VISIBLE_DEVICES=7 python bart_dct/summarize.py \
	--model-dir  ${CHECKPOINTS_DIR} \
	--model-file checkpoint_best.pt \
	--src /path/to/test.txt \
	--out /path/to/hypo.txt \
	--bsz 1024 \


HYPOTHESES=/path/to/test.txt
REFERENCES=/path/to/hypo.txt
python compute_rouge.py --hypotheses $HYPOTHESES --references $REFERENCES

