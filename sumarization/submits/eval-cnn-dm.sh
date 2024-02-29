#!/usr/bin/env bash
CHECKPOINTS_DIR=/path/to/checkpoint/
cp cnn_dm-bin-large-jushi/dict.source.txt $CHECKPOINTS_DIR
cp cnn_dm-bin-large-jushi/dict.target.txt $CHECKPOINTS_DIR
CUDA_VISIBLE_DEVICES=0 python bart_dct/summarize.py   --model-dir $CHECKPOINTS_DIR   --model-file checkpoint_best.pt   --src /path/to/test.source   --out $CHECKPOINTS_DIR/test.hypo --bsz 4096

export CLASSPATH=/path/to/stanford-corenlp-3.7.0.jar
cat $CHECKPOINTS_DIR/test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $CHECKPOINTS_DIR/test.hypo.tokenized
cat cnn-dailymail/cnn_dm/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $CHECKPOINTS_DIR/test.target.tokenized
files2rouge $CHECKPOINTS_DIR/test.hypo.tokenized $CHECKPOINTS_DIR/test.target.tokenized > $CHECKPOINTS_DIR/rouge.txt