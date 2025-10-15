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


I need a more sustainable set of functions for buildiing my research code. In particular, the first variant of experiment (iteration on every question-answer pair) has already been achieved but not the second one. To modify the second one, the optimization should be able to be applied to either one or multiple.


Variant 1 of experiment 2:
1. Load a question-answer pair
2. Get the reward for the default trajectory. This part is relatively simple...
3. Start iteration for the particular problem. Build the attacker prompt, then generate an attack (by prompting the attacker model)
4. Output a postfix only (add this configuration in build_attacker_prompt, and another one for optimizing multiple prompts at one time...)
    a. Or output a completely changed step 1/question
5. Evaluate the question-answer pair on the PRM
6. Finish evaluating and record data.

Debug variant 1 of experiment 2:
1. Design a visualization: % change in reward as a function of iterations. Just save all the data and make a visualization later.
2. Parallelize the evaluation: try to figure out how to evaluate as quickly as possible
3. Change from prm800k to processbench
4. Try both (postfix only) and (free to change step however)
5. Refactor the code


Once the optimal prompt is determined, we can continue to use it. Especially for batches



(8373, 'gsm8k-245', -1, 1, "Mark is a copy-editor. He edits an equal number of sentences each week for two different publishers, who each pay him a different rate per sentence. Publisher B pays Mark twice what Publisher A pays. Mark edits a total number of 1000 sentences each week, but it's almost certain that he edits 1000 sentences for one publisher and 0 sentences for the other publisher, each week. Publisher A pays him 5 cents per sentence. How much does Mark make in a week, in cents?", '[0.2619722783565521, 0.26175495982170105, 0.3304426074028015, 0.3189254701137543, 0.39180564880371094, 0.3329062759876251]', 'catattack iteration 0', '10fa5fefa11793773665b1f7024f2adffa65c495')

(9788, 'gsm8k-46', -1, 1, "While Joanne is gathering apples from her family’s orchard, her sister comes outside to help her. Joanne gathers 30 
apples from the tallest trees, half this amount from the shortest trees, and more apples from the average trees. Compared with Joanne, her sister gathers twice as many apples from the tallest trees and 3 times as many apples from the shortest trees. She doesn't take any from the average trees. If the sisters have gathered a combined total of 500 apples, there are a few possible values for x. How many apples did Joanne gather from the average trees?", '[0.1826907992362976, 0.310317724943161, 0.6289856433868408, 0.19973568618297577, 0.15590889751911163]', 'catattack iteration 0', '10fa5fefa11793773665b1f7024f2adffa65c495')

Managed to fix bugs in our code and the algorithm runs fine
But expected results did not appear in general
Show specific examples where the attack was successful
What are the modifications that are coming out?
For the best ones, write a quick script to evaluate them on other entries as well.
Also try plotting the trajectories for unsuccessful attacks as well. What does the change in question look like for both

After the meeting, determine if something's wrong with our code by trying catattack on the original intended purpose (reasoning lms)