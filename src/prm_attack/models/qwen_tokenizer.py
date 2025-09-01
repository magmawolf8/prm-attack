"""Thin helper around AutoTokenizer. Enforces formatting rules for que-
stion-answer sequences, and helps with making inputs and attention mas-
ks for both discrete tokens and continuous embeddings."""




# tokenizer modules
from transformers import AutoTokenizer, BatchEncoding
# tensor modules
import torch




class SkyworkTokenizerAPI:
    def __init__(self, model_name: str, default_step_token):
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.step_token = default_step_token
        self.step_token_ids = self._tokenizer.encode(default_step_token)

    def prepare_steps(self, questions: str | list[str], 
                      batch_answers_steps: list[str] | list[list[str]]):
        if isinstance(questions, str):
            questions = [questions]
            batch_answers_steps = [batch_answers_steps]
        
        SkyworkTokenizerAPI._check_validity(questions, batch_answers_steps)
        
        batch_input_ids = list()
        batch_answer_index = list()
        batch_reward_indices = list()
        for question, steps in zip(questions, batch_answers_steps):
            input_ids, answer_index, reward_indices = (
                self._encode(question, steps)
            )
            batch_input_ids.append(input_ids)
            batch_answer_index.append(answer_index)
            batch_reward_indices.append(reward_indices)
        
        return self._makeBatchEncoding(
            batch_input_ids, 
            batch_answer_index, 
            batch_reward_indices
        )

    @staticmethod
    def _check_validity(questions, batch_answers_steps):
        if (
            isinstance(questions, list)
            and isinstance(batch_answers_steps, list)
            and all(
                isinstance(steps, list)
                for steps in batch_answers_steps
            )
            and all(
                all(isinstance(s, str) for s in steps)
                for steps in batch_answers_steps
            )
        ):
            if len(questions) != len(batch_answers_steps):
                raise ValueError(
                    f"Batch size mismatch: len(questions): {len(questions)} "
                    f"len(batch_answers_steps): {len(batch_answers_steps)}"
                )
            return

        raise TypeError("Inputs must be (str, list[str]) or "
                        "(list[str], list[list[str]]).")

    def _encode(self, question: str, answer_steps: list[str]):
        """Encodes one question-answer pair.

        Uses the tokenizer and the step token marker to build a token
        sequence and reward flags. Adds a newline between the question
        and the answer, and adds the step token between answer steps.

        Args:
            question: the question string
            answer_steps: a list of answer steps
        
        Returns:
            a tuple of the following data in order:
            1. input_ids: the token sequence.
            2. answer_index: the index of the first token of the answer
                in the token sequence.
            3. reward_indices: a list with the indices of the last tok-
                en of the separator between steps in input_ids.
        """
        question_ids = self._tokenizer.encode(
            self._tokenizer.bos_token + question + '\n'
        )

        answer_index = len(question_ids)

        answer_ids = list()
        reward_indices = list()
        end_of_step = answer_index - 1
        for step in answer_steps:
            step_ids = self._tokenizer.encode(step)
            step_ids.extend(self.step_token_ids)

            end_of_step += len(step_ids)

            answer_ids.extend(step_ids)
            reward_indices.append(end_of_step)
        
        input_ids = question_ids + answer_ids

        return input_ids, answer_index, reward_indices

    def _makeBatchEncoding(self, batch_input_ids: list[list[int]],
                           batch_answer_index: list[int],
                           batch_reward_indices: list[list[int]]):
        """Make a batch encoding object from the given input.
        
        Creates PyTorch tensors by padding the input_ids for each entry
        in the batch, then generates the attention mask and the reward
        flag tensors.

        Args:
            batch_input_ids: input_ids for each entry in the batch.
            batch_reward_indices: reward_indices for each entry, the i-
                ndices of the last token of separator between steps.

        Returns:
            A BatchEncoding object containing the data the model needs.
            Unpack the object in the model call parameters.
        """
        padded_batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(input_ids) for input_ids in batch_input_ids],
            batch_first=True,
            padding_value=self._tokenizer.pad_token_id
        )

        pad_len = padded_batch_input_ids.shape[1]

        batch_attention_mask = list()
        for input_ids in batch_input_ids:
            attention_mask = torch.zeros(pad_len, dtype=torch.long)
            attention_mask[:len(input_ids)] = 1
            batch_attention_mask.append(attention_mask)
        padded_batch_attention_mask = torch.vstack(batch_attention_mask)

        batch_answer_flag = list()
        batch_reward_flags = list()
        for a_ind, r_inds in zip(batch_answer_index, batch_reward_indices):
            answer_flag = torch.zeros(pad_len)
            answer_flag[a_ind] = 1
            reward_flags = torch.zeros(pad_len)
            for index in r_inds:
                reward_flags[index] = 1
            batch_answer_flag.append(answer_flag)
            batch_reward_flags.append(reward_flags)
        padded_batch_answer_flag = torch.vstack(batch_answer_flag)
        padded_batch_reward_flags = torch.vstack(batch_reward_flags)

        return BatchEncoding(
            data={"input_ids": padded_batch_input_ids,
                  "attention_mask": padded_batch_attention_mask,
                  "answer_flag": padded_batch_answer_flag,
                  "reward_flags": padded_batch_reward_flags}
        )