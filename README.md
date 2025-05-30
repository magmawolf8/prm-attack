`git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git ./src/models/skywork_o1_prm_inference`

New changes:
Completed more modular codebase around skywork o1 open PRM. Files:
1. clear_skywork.py: inherits from skywork's nn.module subclass to allow inputs_embeds
2. skywork_tokenizer.py: improved tokenization performance, wrapper for HF tokenizer
3. conftest.py and test_prm.py: ensure that the properties of the PRM are preserved in refactor
4. pyproject.toml: python package for ease of use

Created new statistics.py file designed to estimate the cross-entropy of two probability distributions
    the skywork process reward model (in the space of all question-answer pairs), and the ground truth.
    Estimated using any dataset with "problem," "steps," and "label" (label of first incorrect step) keys
1. Cross-entropy describes how "inefficient" a classification is
2. Wrote simple script demonstrating how to use the prm_attack package to calculate the cross-entropy

Wrote script: simple_sgd.py. (discuss results)


Next steps:
1. Coordinate with Rishabh to incorporate more descriptive statistics pertaining to process reward models.
    1. The method of estimating cross-entropy of the PRM and the ground truth using a sample dataset, might be useful
    2. need another function estimating the entropy of the dataset itself.
    3. Then our job is to maximize the cross-entropy, to make the classification as bad as possible.
2. Then, I'll test statistics methods on the biases which we presented in presentation on 5/19.
3. Try more comprehensive gradient descent methods. In particular:
    1. Constrain gradient optimization to most sensitive tokens?
    2. Prefix randomly-initialized vectors at the start of the answer trajectory, and optimize those.
    3. Create new dataset, adversarially modified to make binary classification as bad as posssible.
4. Need more visualizations in visualization.py (currently empty)