# Copyright (c) Meta Platforms, Inc. and affiliates.
from enum import Enum
from dataclasses import dataclass
from typing import Any, Generator, List, Optional

import torch
# No longer importing numpy directly for core operations,
# but pydantic/BaseModel might use it internally or for type hints if BltSequence uses it.
# from pydantic import BaseModel # Assuming this is defined if BltSequence uses it
# from sequence_iterator import SequenceIterator # Assuming this is defined elsewhere


# --- Placeholder definitions for context ---
# (These would be your actual definitions from your project structure)
class PackingMode(str, Enum):
    BYTES = "bytes"
    PATCHING = "patching"

class PackingArgs:
    def __init__(
        self,
        batch_size: int,
        seq_len: int, # For BYTES: token seq len; For PATCHING: num patches
        pad_id: int,
        max_length: Optional[int], # Max token length for PATCHING mode
        pad_to_max_length: bool,
        enable_byte_ngrams: bool,
        packing_mode: PackingMode,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.enable_byte_ngrams = enable_byte_ngrams
        self.packing_mode = packing_mode

# Assuming BltSequence comes from sequence_iterator or similar
# For testing, let's define a simple version.
# If BltSequence uses pydantic.BaseModel, ensure pydantic is installed.

@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    mask: Optional[torch.Tensor] = None
    patch_lengths: Optional[torch.Tensor] = None
    ngram_ids: Optional[torch.Tensor] = None # Assuming this might also become a tensor
    is_final: bool = False


def truncate_and_rectangularise(
    batch,
    *,
    max_length: int,
    pad_id: int,
    patches_per_seq: int,
    pad_to_max_length: bool = False,
):
    dev = batch.tokens.device
    B = batch.patch_lengths.shape[0] // patches_per_seq
    
    if B == 0:
        width_for_empty = max_length if pad_to_max_length else 0
        empty_shape_data = (0, width_for_empty)
        empty_tensor_int_data = torch.empty(empty_shape_data, dtype=batch.tokens.dtype, device=dev)
        empty_tensor_bool_data = torch.empty(empty_shape_data, dtype=torch.bool, device=dev)
        pl_empty_shape = (0, patches_per_seq)
        empty_pl_tensor = torch.empty(pl_empty_shape, dtype=batch.patch_lengths.dtype, device=dev)
        return empty_tensor_int_data, empty_tensor_int_data, empty_pl_tensor, empty_tensor_bool_data

    pl = batch.patch_lengths.view(B, patches_per_seq)
    M = patches_per_seq

    row_len_hdr = pl.sum(1)
    row_len_tok = row_len_hdr - 1
    row_len_tok_orig = row_len_tok.clone()
    max_hdr = max_length + 1
    rows_exceed = row_len_hdr > max_hdr

    pl_final = pl.clone()

    if torch.any(rows_exceed):
        cumsum = pl_final.cumsum(1)
        idx_ex = (cumsum > max_hdr).int().argmax(1)
        idx_prev = torch.clamp(idx_ex - 1, min=0)
        count_prev = cumsum.gather(1, idx_prev[:, None]).squeeze(1)
        count_prev = torch.where(idx_ex == 0, torch.zeros_like(count_prev), count_prev)
        new_len_for_exceeding_patch = max_hdr - count_prev

        m_axis = torch.arange(M, device=dev)
        kill_mask = (m_axis[None, :] > idx_ex[:, None]) & rows_exceed[:, None]
        pl_final.masked_fill_(kill_mask, 0)
        
        # Apply new_len only to relevant rows and their specific idx_ex
        rows_exceed_indices = torch.where(rows_exceed)[0]
        pl_final[rows_exceed_indices, idx_ex[rows_exceed_indices]] = new_len_for_exceeding_patch[rows_exceed_indices]
        
        row_len_tok = torch.where(
            rows_exceed, torch.full_like(row_len_tok, max_length, device=dev), row_len_tok
        )

    if pad_to_max_length:
        need_pad = row_len_tok < max_length
        if torch.any(need_pad):
            diff = max_length - row_len_tok[need_pad]
            
            # Ensure pl_final[need_pad] is not empty before operating
            if pl_final[need_pad].numel() > 0 :
                last_idx_for_padding = (pl_final[need_pad] != 0).sum(1) - 1
                last_idx_for_padding = torch.clamp(last_idx_for_padding, min=0)
                
                rows_needing_pad_indices = torch.where(need_pad)[0]
                # Ensure last_idx_for_padding is valid for indexing pl_final
                if rows_needing_pad_indices.numel() > 0: # if there are rows to pad
                    pl_final[rows_needing_pad_indices, last_idx_for_padding] += diff
            
            row_len_tok[need_pad] = max_length
    
    width = 0
    if B > 0 and row_len_tok.numel() > 0:
        max_row_len_tok = row_len_tok.max().item()
        width = int(max_row_len_tok) if max_row_len_tok >= 0 else 0
    width = max(0, width) # Ensure width is not negative

    tokens2d = torch.full((B, width), pad_id, dtype=batch.tokens.dtype, device=dev)
    tokens2d_y = torch.full((B, width), pad_id, dtype=batch.tokens.dtype, device=dev)
    mask_x_2d = torch.full((B, width), False, dtype=torch.bool, device=dev)

    if width > 0:
        src_off = torch.cat([torch.tensor([0], device=dev, dtype=batch.source_segment_lengths.dtype), batch.source_segment_lengths.cumsum(dim=0)[:-1]])

        idx_mat = torch.arange(width, device=dev).expand(B, -1) # expand B rows
        copy_len = torch.minimum(row_len_tok, row_len_tok_orig)
        copy_len = torch.clamp(copy_len, min=0)

        valid_copy_mask = idx_mat < copy_len.unsqueeze(1)
        src_flat_idx = src_off.unsqueeze(1) + idx_mat

        if batch.tokens.numel() > 0:
            safe_src_flat_idx = torch.where(valid_copy_mask, src_flat_idx, torch.zeros_like(src_flat_idx))
            gathered_all_potential_tokens = batch.tokens.gather(0, safe_src_flat_idx.view(-1)).view(B, width)
            tokens2d[valid_copy_mask] = gathered_all_potential_tokens[valid_copy_mask]
            mask_x_2d[valid_copy_mask] = True
            src_flat_idx_y = src_off.unsqueeze(1) + idx_mat + 1
            valid_y_source_token_exists = (idx_mat + 1) < batch.source_segment_lengths.unsqueeze(1)
            valid_copy_mask_y = valid_copy_mask & valid_y_source_token_exists
            safe_src_flat_idx_y = torch.where(valid_copy_mask_y, src_flat_idx_y, torch.zeros_like(src_flat_idx_y))
            gathered_y_tokens = batch.tokens.gather(0, safe_src_flat_idx_y.view(-1)).view(B, width)
            tokens2d_y[valid_copy_mask_y] = gathered_y_tokens[valid_copy_mask_y]
    

    if B > 0:
        if pad_to_max_length:
            assert torch.all(pl_final.sum(1) == max_hdr), f"Not all patch sums are {max_hdr}. Sums: {pl_final.sum(1)}"
        else:
            assert torch.all(pl_final.sum(1) <= max_hdr), f"Some patch sums exceed {max_hdr}. Sums: {pl_final.sum(1)}"
    
    assert width <= max_length or (not pad_to_max_length and width == 0), f"Width {width} > max_length {max_length}"
    if pad_to_max_length and B > 0:
        if width > 0 or max_length == 0:
            assert width == max_length, f"Width {width} != max_length {max_length} when pad_to_max_length=True"
            
    return tokens2d, tokens2d_y, pl_final, mask_x_2d


class PackingIterator:
    def __init__(
        self,
        sequence_iterator,
        packing_args: PackingArgs,
    ):
        self.sequence_iterator = sequence_iterator
        self.packing_args = packing_args

    def __iter__(self) -> Generator[Batch, Any, None]:
        if self.packing_args.packing_mode == PackingMode.BYTES:
            yield from self._create_iter_from_bytes()
        elif self.packing_args.packing_mode == PackingMode.PATCHING:
            yield from self._create_iter_from_patch_lengths()
        else:
            raise ValueError(f"Invalid patching mode: {self.packing_args.packing_mode}")

    def _create_iter_from_bytes(self):
        sequence_iter = iter(self.sequence_iterator) # Expects BltSequence-like objects
        batch_size = self.packing_args.batch_size
        pad_id = self.packing_args.pad_id
        target_token_seq_len = self.packing_args.seq_len # Max token length for batch items

        while True:
            tokens_list: List[List[int]] = []
            masks_list: List[List[bool]] = []
            stop_iteration = False
            try:
                for _ in range(batch_size):
                    sequence = next(sequence_iter)
                    _tokens = sequence.tokens
                    _mask = sequence.mask
                    assert (
                        sequence.patch_lengths is None
                    ), "patch_lengths should not be used in byte packing"
                    tokens_list.append(_tokens)
                    masks_list.append(_mask)
            except StopIteration:
                stop_iteration = True

            if not tokens_list and stop_iteration:
                break
            
            current_batch_size = len(tokens_list)
            if current_batch_size == 0 and stop_iteration:
                 break

            # device can be set globally or inferred, e.g. torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # For simplicity, using default device (CPU) or allowing Batch.from_python_dict to handle it.
            # If tensors need to be on a specific device, ensure it here.
            current_device = torch.device("cpu") # Example, can be configurable

            x = torch.full((batch_size, target_token_seq_len), fill_value=pad_id, dtype=torch.int64, device=current_device)
            y = torch.full((batch_size, target_token_seq_len), fill_value=pad_id, dtype=torch.int64, device=current_device)
            m_x = torch.zeros((batch_size, target_token_seq_len), dtype=torch.bool, device=current_device)

            for i in range(current_batch_size):
                tok_seq = tokens_list[i]
                mask_seq = masks_list[i]
                len_tok_seq = len(tok_seq)
                
                if len_tok_seq > 0:
                    x[i, :len_tok_seq] = torch.tensor(tok_seq, dtype=torch.int64, device=current_device)
                    if len_tok_seq > 1:
                        y[i, :len_tok_seq - 1] = torch.tensor(tok_seq[1:], dtype=torch.int64, device=current_device)
                    m_x[i, :len_tok_seq] = torch.tensor(mask_seq[:len_tok_seq], dtype=torch.bool, device=current_device)

            output_batch = Batch(x=x, y=y, mask=m_x)
            assert (
                output_batch.mask is None or torch.sum(x != pad_id).item() == torch.sum(output_batch.mask).item()
            ), f"BYTES mode: {torch.sum(x != pad_id).item()} != {torch.sum(output_batch.mask).item()}"
            
            yield output_batch

            if stop_iteration:
                break

    def _create_iter_from_patch_lengths(self):
        batch_size = self.packing_args.batch_size
        pad_id = self.packing_args.pad_id
        num_patches_m = self.packing_args.seq_len # seq_len is num_patches
        pad_to_max_length = self.packing_args.pad_to_max_length
        max_token_len = self.packing_args.max_length
        
        assert max_token_len is not None, "max_length must be provided for patch-based packing"

        running_toks_x: Optional[torch.Tensor] = None
        running_toks_y: Optional[torch.Tensor] = None
        running_patch_lengths: Optional[torch.Tensor] = None
        running_masks_x: Optional[torch.Tensor] = None

        for chunk_from_seq_iter in self.sequence_iterator.create_iter():
            if chunk_from_seq_iter.tokens.numel() == 0:
                continue

            dev = chunk_from_seq_iter.tokens.device
            
            num_seqs_in_chunk = chunk_from_seq_iter.patch_lengths.shape[0] // num_patches_m
            if num_seqs_in_chunk == 0:
                continue

            pl_2d_orig = chunk_from_seq_iter.patch_lengths.view(num_seqs_in_chunk, num_patches_m)
            
            original_p0s = pl_2d_orig[:, 0].clone()
            transformed_pl = pl_2d_orig.clone()
            needs_transform = original_p0s > 1 # Boolean tensor for rows needing transformation

            if num_patches_m == 1:
                transformed_pl[needs_transform, 0] = 1
            elif num_patches_m > 1:
                # Apply transformation only to rows where needs_transform is True
                rows_to_transform_indices = torch.where(needs_transform)[0]
                if rows_to_transform_indices.numel() > 0:
                    transformed_pl[rows_to_transform_indices, 0] = 1
                    transformed_pl[rows_to_transform_indices, 1] = original_p0s[rows_to_transform_indices] - 1
                    if num_patches_m > 2: # If there's space for p1, p2, ...
                        # pl_2d_orig[rows, 1:-1] copies p1, ..., p(M-2) from original
                        transformed_pl[rows_to_transform_indices, 2:] = pl_2d_orig[rows_to_transform_indices, 1:-1]
            
            # Create a temporary object for truncate_and_rectangularise
            batch_for_tnr = type('BatchForTNR', (), {})() 
            batch_for_tnr.tokens = chunk_from_seq_iter.tokens
            batch_for_tnr.patch_lengths = transformed_pl.view(-1) # Flatten back
            batch_for_tnr.masks_flat = getattr(chunk_from_seq_iter, 'masks_flat', None)
            batch_for_tnr.source_segment_lengths = pl_2d_orig.sum(dim=1)


            tokens_x, tokens_y, final_pl_2d, masks_x = truncate_and_rectangularise(
                batch_for_tnr,
                max_length=max_token_len,
                pad_id=pad_id,
                patches_per_seq=num_patches_m,
                pad_to_max_length=pad_to_max_length,
            )
            
            # Ensure tensors are on the correct device (same as input chunk or a target device)
            tokens_x, tokens_y, final_pl_2d, masks_x = (
                tokens_x.to(dev), tokens_y.to(dev), final_pl_2d.to(dev), masks_x.to(dev)
            )

            if running_toks_x is not None:
                tokens_x = torch.cat((running_toks_x.to(dev), tokens_x), dim=0)
                tokens_y = torch.cat((running_toks_y.to(dev), tokens_y), dim=0)
                final_pl_2d = torch.cat((running_patch_lengths.to(dev), final_pl_2d), dim=0)
                masks_x = torch.cat((running_masks_x.to(dev), masks_x), dim=0)

            while tokens_x.shape[0] >= batch_size:
                batch_x_slice = tokens_x[:batch_size]
                batch_y_slice = tokens_y[:batch_size]
                batch_pl_slice = final_pl_2d[:batch_size]
                batch_mask_x_slice = masks_x[:batch_size]
                
                output_batch = Batch(
                    x=batch_x_slice, y=batch_y_slice, 
                    mask=batch_mask_x_slice, patch_lengths=batch_pl_slice
                )

                yield output_batch

                tokens_x = tokens_x[batch_size:]
                tokens_y = tokens_y[batch_size:]
                final_pl_2d = final_pl_2d[batch_size:]
                masks_x = masks_x[batch_size:]
            
            running_toks_x = tokens_x if tokens_x.numel() > 0 and tokens_x.shape[0] > 0 else None
            running_toks_y = tokens_y if tokens_y.numel() > 0 and tokens_y.shape[0] > 0 else None
            running_patch_lengths = final_pl_2d if final_pl_2d.numel() > 0 and final_pl_2d.shape[0] > 0 else None
            running_masks_x = masks_x if masks_x.numel() > 0 and masks_x.shape[0] > 0 else None
        
        # NOTE: might interfere with model forward pass. various tests were implemented in serialized iterators to force batch size.
        # if running_toks_x is not None and running_toks_x.shape[0] > 0:
        #     num_remaining = running_toks_x.shape[0]
        #     dev = running_toks_x.device
            
        #     final_x_tensor = running_toks_x
        #     final_y_tensor = running_toks_y
        #     final_pl_tensor = running_patch_lengths
        #     final_mask_y_tensor = running_masks_y # This is y-mask

        #     if num_remaining < batch_size:
        #         pad_rows = batch_size - num_remaining
                
        #         pad_x = torch.full((pad_rows, final_x_tensor.shape[1]), pad_id, dtype=final_x_tensor.dtype, device=dev)
        #         final_x_tensor = torch.cat((final_x_tensor, pad_x), dim=0)

        #         pad_y = torch.full((pad_rows, final_y_tensor.shape[1]), pad_id, dtype=final_y_tensor.dtype, device=dev)
        #         final_y_tensor = torch.cat((final_y_tensor, pad_y), dim=0)

        #         pad_pl = torch.zeros((pad_rows, final_pl_tensor.shape[1]), dtype=final_pl_tensor.dtype, device=dev)
        #         if num_patches_m > 0 and max_token_len is not None:
        #             if final_pl_tensor.shape[1] > 0: # Check if there are any patch columns
        #                 pad_pl[:, 0] = max_token_len + 1 # Sum of patches is max_token_len + EOS
        #         final_pl_tensor = torch.cat((final_pl_tensor, pad_pl), dim=0)

        #         pad_mask_y = torch.zeros((pad_rows, final_mask_y_tensor.shape[1]), dtype=torch.bool, device=dev)
        #         final_mask_y_tensor = torch.cat((final_mask_y_tensor, pad_mask_y), dim=0)

        #     output_batch = Batch(
        #         x=final_x_tensor, y=final_y_tensor, 
        #         mask=final_mask_y_tensor, patch_lengths=final_pl_tensor, # mask is y-mask
        #         is_final=True
        #     )
        #     assert (output_batch.mask is None or torch.sum(final_x_tensor != pad_id).item() == torch.sum(output_batch.mask).item()), \
        #         f"Final PATCHING batch assertion: {torch.sum(final_x_tensor != pad_id).item()} != {torch.sum(output_batch.mask).item()}"
        #     yield output_batch
