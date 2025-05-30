# todo list
1. Implement converting mixtures of tokens and input embeddings into total input embeddings
2. Implement tokenizer helper according to docstring
3. update test_prm.py to use tokenizer and prm
4. determine which things we don't need to track gradients on, if performance is suffering

Thinking about how to hijack the PRM. Probably start with the placement of the tokens. You want the tokens to have the greatest impact on the reward, so should I insert extra tokens at the start of each step?

Modifying all the tokens doesn't really do anything, the adversarial change is too fine-grained and the logits just project back to the original tokens.
So we have a few options:
1. Prepend vectors onto the thing with random initialization. Then optimize in that constrained way.
2. Constrain optimization to those with large gradient
3. Try to find the global maximum for some selected tokens with something other than SGD
4. Analyze correlation between ground truth and predicted success probability
5. Or modify all the tokens as before, and see how much higher the reward is across the entire dataset after projecting back to tokens
6. Either way, I'll probably have to write functions to prepend stuff to the answer. Put those in the tokenizer API?