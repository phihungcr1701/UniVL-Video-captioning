import torch
import time
import random
from tqdm import tqdm


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler,
                global_step, logger, auxiliary_combine_net=None, meta_optimizer=None, local_rank=0):
    torch.cuda.empty_cache()
    model.train()
    if auxiliary_combine_net is not None:
        auxiliary_combine_net.eval()
    
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    batchs = []

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        # MELTR: Call with tasks parameter
        if hasattr(args, 'tasks'):
            losses = model(input_ids, segment_ids, input_mask, video, video_mask,
                         pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                         masked_video=masked_video, video_labels_index=video_labels_index,
                         input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                         output_caption_ids=pairs_output_caption_ids, tasks=args.tasks)
            
            # MELTR: Combine losses using auxiliary network
            if auxiliary_combine_net is not None:
                loss = auxiliary_combine_net(torch.stack(losses).unsqueeze(1))
            else:
                loss = sum(losses)
        else:
            loss = model(input_ids, segment_ids, input_mask, video, video_mask,
                         pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                         masked_video=masked_video, video_labels_index=video_labels_index,
                         input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                         output_caption_ids=pairs_output_caption_ids)

        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss.detach() if isinstance(loss, torch.Tensor) else loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            
            # MELTR: Meta-optimizer update (auxiliary network parameters)
            if meta_optimizer is not None and (global_step % args.auxgrad_every == 0):
                if len(batchs) > 0:
                    # Sample a batch from history
                    sample_batch = random.choices(batchs)[0]
                    input_ids_val, input_mask_val, segment_ids_val, video_val, video_mask_val, \
                    pairs_masked_text_val, pairs_token_labels_val, masked_video_val, video_labels_index_val, \
                    pairs_input_caption_ids_val, pairs_decoder_mask_val, pairs_output_caption_ids_val = sample_batch
                    
                    # Compute validation loss
                    losses_val = model(input_ids_val, segment_ids_val, input_mask_val, video_val, video_mask_val,
                                     pairs_masked_text=pairs_masked_text_val, pairs_token_labels=pairs_token_labels_val,
                                     masked_video=masked_video_val, video_labels_index=video_labels_index_val,
                                     input_caption_ids=pairs_input_caption_ids_val, decoder_mask=pairs_decoder_mask_val,
                                     output_caption_ids=pairs_output_caption_ids_val, tasks=args.target_tasks)
                    meta_val_loss = sum(losses_val)
                    
                    # Compute current training loss
                    losses_train = model(input_ids, segment_ids, input_mask, video, video_mask,
                                       pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                                       masked_video=masked_video, video_labels_index=video_labels_index,
                                       input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                                       output_caption_ids=pairs_output_caption_ids, tasks=args.tasks)
                    inner_loop_train_loss = auxiliary_combine_net(torch.stack(losses_train).unsqueeze(1))
                    
                    # Update meta-optimizer
                    phi = list(auxiliary_combine_net.parameters())
                    W = [p for n, p in model.named_parameters()]

                    if args.reg:
                        meta_val_loss += args.gamma * torch.norm(sum(losses_train) - inner_loop_train_loss)

                    meta_optimizer.step(
                        val_loss=meta_val_loss,
                        train_loss=inner_loop_train_loss,
                        aux_params=phi,
                        parameters=W,
                    )
            
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()
        
        # MELTR: Store batch for meta-learning
        if auxiliary_combine_net is not None:
            batchs.append(batch)
            if len(batchs) > 10:
                batchs.pop(0)

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step
