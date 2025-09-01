#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python3 test.py --jsonl phase2_train.jsonl --vllm-base-url http://localhost:12341/v1 --vllm-api-key EMPTY
