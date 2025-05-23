# todo list
1. Implement converting mixtures of tokens and input embeddings into total input embeddings
2. Implement tokenizer helper according to docstring
3. update test_prm.py to use tokenizer and prm
4. determine which things we don't need to track gradients on, if performance is suffering

Thinking about how to hijack the PRM. Probably start with the placement of the tokens. You want the tokens to have the greatest impact on the reward, so should I insert extra tokens at the start of each step?
