from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import random
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import Dataset


# ImageNet normalisation constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_frames_ffmpeg(video_path, start_time, end_time, fps, height, width):
    """Decode video frames with ffmpeg.

    Returns
    -------
    numpy.ndarray, shape (T, 3, H, W), dtype float32
        Frames normalised with ImageNet mean/std.  Returns empty array on
        failure.
    """
    try:
        import ffmpeg
        duration = max(end_time - start_time, 1.0 / fps)
        out, _ = (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .filter('fps', fps=fps)
            .filter('scale', width, height)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        frames = np.frombuffer(out, np.uint8).reshape(-1, height, width, 3)
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - _IMAGENET_MEAN) / _IMAGENET_STD   # (T, H, W, 3)
        frames = frames.transpose(0, 3, 1, 2)                 # (T, 3, H, W)
        return frames
    except Exception as e:
        print("Failed to load {}: {}".format(video_path, e))
        return np.zeros((0, 3, height, width), dtype=np.float32)


class Youcook_Caption_RawVideo_DataLoader(Dataset):
    """YoucookII caption dataset that loads raw video frames at runtime.

    Instead of consuming pre-extracted S3D pickle features, this loader
    decodes video clips on-the-fly using ffmpeg and returns raw RGB frames
    so that UNet3D can extract features end-to-end during training.

    The returned ``video`` tensor has shape ``(1, max_frames, 3, H, W)``
    where H = ``clip_height`` and W = ``clip_width``.  The extra leading
    dimension mirrors the ``k=1`` convention used by the pickle-based loader,
    keeping the rest of the pipeline unchanged.
    """

    def __init__(
            self,
            csv,
            data_path,
            video_path,
            tokenizer,
            feature_framerate=1.0,
            max_words=30,
            max_frames=100,
            clip_height=224,
            clip_width=224,
    ):
        self.csv = pd.read_csv(csv)
        self.data_dict = pickle.load(open(data_path, 'rb'))
        self.video_path = video_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.clip_height = clip_height
        self.clip_width = clip_width
        self.tokenizer = tokenizer

        # Build index: video_id → csv row index
        video_id_list = [itm for itm in self.csv['video_id'].values]
        self.video_id2idx_dict = {vid: i for i, vid in enumerate(video_id_list)}

        # Build flat iteration index: int → (video_id, sub_id)
        self.iter2video_pairs_dict = {}
        iter_idx_ = 0
        for video_id in video_id_list:
            n_caption = len(self.data_dict[video_id]['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[iter_idx_] = (video_id, sub_id)
                iter_idx_ += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        data_dict = self.data_dict[video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k)
        ends = np.zeros(k)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.int64)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int64)

        for i in range(k):
            ind = r_ind[i]
            start_, end_ = data_dict['start'][ind], data_dict['end'][ind]
            starts[i], ends[i] = start_, end_
            total_length_with_CLS = self.max_words - 1
            words = self.tokenizer.tokenize(data_dict['transcript'][ind])

            words = ["[CLS]"] + words
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                else:
                    token_labels.append(-1)

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # Caption tokens for decoder
            caption_words = self.tokenizer.tokenize(data_dict['text'][ind])
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return (pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels,
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends)

    def _get_video_raw(self, video_file, s, e):
        """Return raw video frames and associated mask/label tensors.

        Returns
        -------
        video : ndarray (k, max_frames, 3, H, W)  float32
        video_mask : ndarray (k, max_frames)       int64
        masked_video : ndarray (k, max_frames, 3, H, W)  float32
        video_labels_index : ndarray (k, max_frames)     int64
        """
        k = len(s)
        video = np.zeros(
            (k, self.max_frames, 3, self.clip_height, self.clip_width),
            dtype=np.float32,
        )
        video_mask = np.zeros((k, self.max_frames), dtype=np.int64)
        max_video_length = [0] * k

        video_path = os.path.join(self.video_path, video_file)

        for i in range(k):
            frames = _load_frames_ffmpeg(
                video_path, s[i], e[i],
                self.feature_framerate, self.clip_height, self.clip_width,
            )
            if frames.shape[0] > self.max_frames:
                frames = frames[:self.max_frames]
            n_frames = frames.shape[0]
            max_video_length[i] = n_frames
            if n_frames > 0:
                video[i][:n_frames] = frames

        for i, v_len in enumerate(max_video_length):
            video_mask[i][:v_len] = 1

        # Mask Frame Model: randomly zero-out ~15 % of valid frames
        video_labels_index = np.full((k, self.max_frames), -1, dtype=np.int64)
        masked_video = video.copy()
        for i in range(k):
            for j in range(max_video_length[i]):
                prob = random.random()
                if prob < 0.15:
                    masked_video[i][j] = 0.0
                    video_labels_index[i][j] = j

        return video, video_mask, masked_video, video_labels_index

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, feature_idx):
        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        idx = self.video_id2idx_dict[video_id]

        (pairs_text, pairs_mask, pairs_segment,
         pairs_masked_text, pairs_token_labels,
         pairs_input_caption_ids, pairs_decoder_mask,
         pairs_output_caption_ids, starts, ends) = self._get_text(video_id, sub_id)

        video_file = self.csv["video_file"].values[idx]
        video, video_mask, masked_video, video_labels_index = self._get_video_raw(
            video_file, starts, ends,
        )

        return (pairs_text, pairs_mask, pairs_segment,
                video, video_mask,
                pairs_masked_text, pairs_token_labels,
                masked_video, video_labels_index,
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids)
