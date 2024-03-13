import random
import numpy as np

import torch
import torch.nn as nn
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
from PIL import Image
import models._model_utils as model_utils
import utils


# from utils import correct_alignment
# from utils import create_grid_image

# from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Model  # TODO Remove
# from transformers.models.layoutlmv3.processing_layoutlmv3 import LayoutLMv3Processor    # TODO Remove


class LayoutLMv3_hy:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.apply_ocr = config['apply_ocr']
        self.processor = LayoutLMv3Processor.from_pretrained(config['model_weights'], apply_ocr=config['apply_ocr'])  # Check that this do not fuck up the code.
        # self.processor = LayoutLMv3Processor.from_pretrained(config['model_weights'], apply_ocr=False)  # Check that this do not fuck up the code.
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(config['model_weights'])
        self.ignore_index = 9999  # 0

    # def parallelize(self):
    #     self.model = nn.DataParallel(self.model)

    def forward(self, batch, return_pred_answer=False):

        # bs = len(batch['question_id'])
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']
        images = batch['images']

        boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes']]  # Scale boxes 0->1 to 0-->1000.
        
        if self.apply_ocr:
            encoding = self.processor(images, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        else:
            encoding = self.processor(images, question, batch["words"], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
        outputs = self.model(**encoding, start_positions=start_pos, end_positions=end_pos)
        pred_answers, answ_confidence = self.get_answer_from_model_output(encoding.input_ids, outputs) if return_pred_answer else None

        return outputs, pred_answers, answ_confidence

    def get_concat_v_multi_resize(self, im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample) for im in im_list]

        # Fix equal height for all images (breaks the aspect ratio).
        heights = [im.height for im in im_list]
        im_list_resize = [im.resize((im.height, max(heights)), resample=resample) for im in im_list_resize]

        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    def get_start_end_idx(self, encoding, context, answers):
        pos_idx = []
        for batch_idx in range(len(encoding.input_ids)):
            answer_pos = []
            for answer in answers[batch_idx]:
                encoded_answer = [token for token in self.processor.tokenizer.encode([answer], boxes=[0, 0, 0, 0]) if token not in self.processor.tokenizer.all_special_ids]
                answer_tokens_length = len(encoded_answer)

                for token_pos in range(len(encoding.input_ids[batch_idx])):
                    if encoding.input_ids[batch_idx][token_pos: token_pos+answer_tokens_length].tolist() == encoded_answer:
                        answer_pos.append([token_pos, token_pos + answer_tokens_length-1])

            if len(answer_pos) == 0:
                pos_idx.append([self.ignore_index, self.ignore_index])

            else:
                answer_pos = random.choice(answer_pos)  # To add variability, pick a random correct span.
                pos_idx.append(answer_pos)

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)

        return start_idxs, end_idxs

    def get_answer_from_model_output(self, input_tokens, outputs):
        predicted_start_idxs = torch.argmax(outputs.start_logits, axis=1)
        predicted_end_idxs = torch.argmax(outputs.end_logits, axis=1)

        predicted_answers = [self.processor.tokenizer.decode(input_tokens[batch_idx][predicted_start_idxs[batch_idx]: predicted_end_idxs[batch_idx]+1], skip_special_tokens=True).strip() for batch_idx in range(len(input_tokens))]
        # answers_conf = ((outputs.start_logits.max(dim=1).values + outputs.end_logits.max(dim=1).values) / 2).tolist()

        start_logits = outputs.start_logits.softmax(dim=1).detach().cpu()
        end_logits = outputs.end_logits.softmax(dim=1).detach().cpu()
        answ_confidence = []
        for batch_idx in range(len(input_tokens)):
            conf_mat = np.matmul(np.expand_dims(start_logits[batch_idx].unsqueeze(dim=0), -1),
                                 np.expand_dims(end_logits[batch_idx].unsqueeze(dim=0), 1)).squeeze(axis=0)

            answ_confidence.append(
                conf_mat[predicted_start_idxs[batch_idx], predicted_end_idxs[batch_idx]].item()
            )

        answ_confidence = model_utils.get_extractive_confidence(outputs)

        return predicted_answers, answ_confidence
