import torch
from torch import nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class _LoadedHFVisionTower:
    def __init__(self, vision_tower, image_processor):
        self.vision_tower = vision_tower
        self.image_processor = image_processor
        self.is_loaded = True

    def load_model(self, device_map=None):
        return None

    def to(self, *args, **kwargs):
        self.vision_tower.to(*args, **kwargs)
        return self

    def __call__(self, *args, **kwargs):
        return self.vision_tower(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.vision_tower, name)


class LlavaLearnablePruneOfficialAdapter(nn.Module):
    def __init__(self, model, image_processor=None):
        super().__init__()
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        self.model = model
        self.image_processor = image_processor

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def get_vision_tower(self):
        return _LoadedHFVisionTower(self.model.vision_tower, self.image_processor)

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def _expand_image_token_inputs(self, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None or not torch.is_tensor(input_ids) or not (input_ids == IMAGE_TOKEN_INDEX).any():
            return kwargs

        image_token_id = int(getattr(self.model.config, "image_token_id", getattr(self.model.config, "image_token_index", -1)))
        image_seq_length = int(getattr(self.model.config, "image_seq_length", 1))
        if image_token_id < 0 or image_seq_length <= 1:
            input_ids = input_ids.clone()
            input_ids[input_ids == IMAGE_TOKEN_INDEX] = image_token_id
            kwargs["input_ids"] = input_ids
            return kwargs

        attention_mask = kwargs.get("attention_mask")
        labels = kwargs.get("labels")
        pad_token_id = int(getattr(self.model.config, "pad_token_id", 0) or 0)

        expanded_ids = []
        expanded_masks = [] if attention_mask is not None else None
        expanded_labels = [] if labels is not None else None

        for row_idx, row in enumerate(input_ids):
            row_ids = []
            row_mask = [] if attention_mask is not None else None
            row_labels = [] if labels is not None else None
            for col_idx, token in enumerate(row):
                token_value = int(token.item())
                if token_value == IMAGE_TOKEN_INDEX:
                    row_ids.extend([token.new_tensor(image_token_id)] * image_seq_length)
                    if row_mask is not None:
                        row_mask.extend([attention_mask[row_idx, col_idx]] * image_seq_length)
                    if row_labels is not None:
                        row_labels.extend([labels.new_tensor(IGNORE_INDEX)] * image_seq_length)
                else:
                    row_ids.append(token)
                    if row_mask is not None:
                        row_mask.append(attention_mask[row_idx, col_idx])
                    if row_labels is not None:
                        row_labels.append(labels[row_idx, col_idx])
            expanded_ids.append(torch.stack(row_ids))
            if expanded_masks is not None and row_mask is not None:
                expanded_masks.append(torch.stack(row_mask))
            if expanded_labels is not None and row_labels is not None:
                expanded_labels.append(torch.stack(row_labels))

        max_len = max(row.shape[0] for row in expanded_ids)

        def _pad_rows(rows, pad_value):
            padded = []
            for row in rows:
                if row.shape[0] < max_len:
                    pad = row.new_full((max_len - row.shape[0],), pad_value)
                    row = torch.cat([row, pad], dim=0)
                padded.append(row)
            return torch.stack(padded, dim=0)

        kwargs["input_ids"] = _pad_rows(expanded_ids, pad_token_id)
        if expanded_masks is not None:
            kwargs["attention_mask"] = _pad_rows(expanded_masks, 0)
        if expanded_labels is not None:
            kwargs["labels"] = _pad_rows(expanded_labels, IGNORE_INDEX)
        return kwargs

    def _normalize_llava_kwargs(self, kwargs):
        if "images" in kwargs and kwargs.get("pixel_values") is None:
            kwargs["pixel_values"] = kwargs.pop("images")
        else:
            kwargs.pop("images", None)
        if getattr(self.model.config, "image_aspect_ratio", None) != "anyres":
            kwargs.pop("image_sizes", None)
        pixel_values = kwargs.get("pixel_values")
        if torch.is_tensor(pixel_values):
            kwargs["pixel_values"] = pixel_values.to(device=self.device, dtype=self.dtype)
        elif isinstance(pixel_values, list):
            kwargs["pixel_values"] = [
                value.to(device=self.device, dtype=self.dtype) if torch.is_tensor(value) else value
                for value in pixel_values
            ]
        return self._expand_image_token_inputs(kwargs)

    def forward(self, *args, **kwargs):
        if args:
            kwargs["input_ids"] = args[0]
            args = args[1:]
        return self.model(*args, **self._normalize_llava_kwargs(kwargs))

    def _slice_generated_tokens(self, outputs, prompt_length):
        if prompt_length is None:
            return outputs
        if torch.is_tensor(outputs):
            if outputs.ndim >= 2 and outputs.shape[-1] >= prompt_length:
                return outputs[:, prompt_length:]
            return outputs

        sequences = getattr(outputs, "sequences", None)
        if torch.is_tensor(sequences) and sequences.ndim >= 2 and sequences.shape[-1] >= prompt_length:
            outputs.sequences = sequences[:, prompt_length:]
        return outputs

    def generate(self, *args, **kwargs):
        if args:
            kwargs["input_ids"] = args[0]
            args = args[1:]
        kwargs = self._normalize_llava_kwargs(kwargs)
        input_ids = kwargs.get("input_ids")
        prompt_length = input_ids.shape[-1] if torch.is_tensor(input_ids) else None
        outputs = self.model.generate(*args, **kwargs)
        return self._slice_generated_tokens(outputs, prompt_length)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
