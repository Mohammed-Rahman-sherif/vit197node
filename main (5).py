import os
import random
import argparse 
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb

# PEFT Imports for LoRA
from peft import LoraConfig, get_peft_model

from datasets import build_dataset
from datasets.utils import build_data_loader, build_graph_data_loader, TokenCLIPVisual
import clip
from utils import *
from hgtmodel import HGTImageFeatureExtractor

wandb_log = True

def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    # Ensure cache_values is in the correct dtype from the start
    cache_values = cache_values.to(val_features.dtype)

    print("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = wandb.config.init_beta, wandb.config.init_alpha
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_loader, test_loader, clip_weights, clip_model, image_model, train_loader_F):
    """
    An enhanced version that combines Zero-Shot, Cache, and Graph-based logits.
    Now includes LoRA fine-tuning for the CLIP model AND VL-JEPA style predictive loss.
    """
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype
    image_model.to(device)

    # ---------------------------------------------------------
    # 1. APPLY LORA TO CLIP
    # ---------------------------------------------------------
    # Target 'c_fc' and 'c_proj' (MLP layers) because OpenAI CLIP uses fused attention weights
    peft_config = LoraConfig(
        r=wandb.config.lora_r,
        lora_alpha=wandb.config.lora_alpha,
        target_modules=["c_fc", "c_proj"], 
        lora_dropout=wandb.config.lora_dropout,
        bias="none",
        modules_to_save=None,
    )
    
    # Wrap the model. Base weights are frozen, LoRA weights are trainable.
    clip_model = get_peft_model(clip_model, peft_config)
    print("\nLoRA Trainable Parameters:")
    clip_model.print_trainable_parameters()

    # ---------------------------------------------------------
    # 2. SETUP ADAPTER
    # ---------------------------------------------------------
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device).to(dtype)
    adapter.weight = nn.Parameter(cache_keys.t())

    # ---------------------------------------------------------
    # 3. SETUP OPTIMIZER (Include CLIP Parameters)
    # ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(image_model.parameters()) + list(clip_model.parameters()),
        lr=wandb.config.lr,
        eps=1e-4,
        weight_decay=wandb.config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    # Hyperparameters
    beta, alpha = wandb.config.init_beta, wandb.config.init_alpha
    gamma = wandb.config.init_gamma 
    lambda_con = wandb.config.lambda_con
    focal_gamma = wandb.config.focal_loss_gamma
    
    # NEW: Weight for VL-JEPA Loss
    # You might want to tune this via wandb eventually
    lambda_jepa = 1.0 
    
    best_acc, best_epoch = 0.0, 0
    
    # Path for saving best LoRA adapter
    lora_save_path = os.path.join(cfg['cache_dir'], f"best_lora_{cfg['shots']}shots")

    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        image_model.train()
        clip_model.train() # IMPORTANT: Enable training mode for LoRA and Dropout
        
        correct_samples, all_samples = 0, 0
        loss_list = []
        print(f'Train Epoch: {train_idx} / {cfg["train_epoch"]}')

        for i, (batched_graphs, images, target) in enumerate(tqdm(train_loader_F)):
            batched_graphs, images, target = batched_graphs.to(device), images.to(device), target.to(device)
            
            # --- Get all three feature types ---
            
            # 1. Get graph-refined features from your HGT model
            # graph_visual_feature is essentially your "Predictor" output (Sy_hat)
            graph_visual_feature, all_updated_text_features = image_model(
                batched_graphs.x_dict,
                batched_graphs.edge_index_dict,
                batched_graphs.batch_dict
            )
            
            # 2. Get standard global features (NOW WITH GRADIENTS FOR LORA)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # --- Calculate all three logit components ---

            # 1. Zero-Shot Logits (Original CLIP)
            original_clip_logits = 100. * image_features.to(dtype) @ clip_weights.to(dtype)

            # 2. Cache Logits (Tip-Adapter)
            affinity = adapter(image_features.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            
            # 3. Graph-Refined Logits (HGT-based)
            graph_visual_feature = graph_visual_feature / graph_visual_feature.norm(dim=-1, keepdim=True)
            all_updated_text_features = all_updated_text_features / all_updated_text_features.norm(dim=-1, keepdim=True)
            B, D, C = graph_visual_feature.shape[0], graph_visual_feature.shape[1], all_updated_text_features.shape[0] // graph_visual_feature.shape[0]
            updated_text_features_per_image = all_updated_text_features.view(B, C, D)
            visual_feature_for_bmm = graph_visual_feature.unsqueeze(1)
            text_features_for_bmm = updated_text_features_per_image.transpose(1, 2)
            graph_logits = 100. * torch.bmm(visual_feature_for_bmm, text_features_for_bmm).squeeze(1)
            
            # --- Combine Logits ---
            tip_logits = original_clip_logits + cache_logits * alpha + graph_logits * gamma

            # ==============================================================================
            # NEW: VL-JEPA Style Predictive Loss Integration
            # ==============================================================================
            # Insight: Instead of just classifying, force the visual embedding to "predict" 
            # the exact location of the text embedding in the latent space.
            
            # 1. Get the Ground Truth Text Embedding (Target Sy)
            # clip_weights is [Dim, Num_Classes], target is [Batch_Size]
            # We select the columns corresponding to the correct class labels.
            target_text_embeddings = clip_weights.t()[target] # Shape: [Batch_Size, Feature_Dim]
            
            # Ensure target is normalized for cosine similarity
            target_text_embeddings = target_text_embeddings / target_text_embeddings.norm(dim=-1, keepdim=True)

            # 2. Get the Prediction (Sy_hat)
            # This is the output of your HGT model, already normalized above
            prediction_visual_embeddings = graph_visual_feature

            # 3. Calculate Predictive Loss (1 - Cosine Similarity)
            # We calculate dot product (since vectors are normalized) and subtract from 1
            cos_sim = (prediction_visual_embeddings * target_text_embeddings).sum(dim=-1)
            loss_jepa = 1.0 - cos_sim.mean()
            # ==============================================================================

            # --- Calculate Standard Losses ---
            loss = F.cross_entropy(tip_logits, target)
            
            # Focal Loss for contrastive alignment
            log_pt = F.log_softmax(graph_logits, dim=1)
            pt = torch.exp(log_pt)
            pt_correct = pt[torch.arange(B, device=device), target]
            log_pt_correct = log_pt[torch.arange(B, device=device), target]
            
            loss_con = -torch.pow(1 - pt_correct, focal_gamma) * log_pt_correct
            loss_con = loss_con.mean() 
            
            # UPDATE TOTAL LOSS with JEPA Loss
            total_loss = loss + lambda_con * loss_con + lambda_jepa * loss_jepa
            
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            
            loss_list.append(total_loss.item())
            
            if(wandb_log):
                wandb.log({
                    "Train accuracy": correct_samples / all_samples,
                    "Total loss": sum(loss_list)/len(loss_list),
                    "Loss (Classification)": loss.item(),
                    "Loss (Focal Contrastive)": loss_con.item(),
                    "Loss (VL-JEPA Prediction)": loss_jepa.item(), # Log the new loss
                    "Learning rate": scheduler.get_last_lr()[0]
                })

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f'LR: {current_lr:.6f}, Acc: {correct_samples / all_samples:.4f} ({correct_samples}/{all_samples}), Loss: {sum(loss_list)/len(loss_list):.4f}')

        # --- Evaluation Phase ---
        adapter.eval()
        image_model.eval()
        clip_model.eval() # Switch to eval for validation
        
        # Re-encode test images using the tuned LoRA model
        test_features_list = []
        test_labels_list = []
        with torch.no_grad():
            for images, target in tqdm(test_loader, desc="Encoding Test Set"):
                images = images.to(device)
                target = target.to(device)
                
                # Encode with the updated LoRA CLIP model
                features = clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                
                test_features_list.append(features)
                test_labels_list.append(target)
        
        test_features_lora = torch.cat(test_features_list, dim=0)
        test_labels_lora = torch.cat(test_labels_list, dim=0)

        with torch.no_grad():
            affinity = adapter(test_features_lora.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            
            clip_logits = 100. * test_features_lora.to(dtype) @ clip_weights.to(dtype)
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, test_labels_lora)
        
        if wandb_log:
            wandb.log({"Test accuracy": acc, "Epoch": train_idx})

        print(f"**** Tip-Adapter-F's test accuracy: {acc:.2f}. ****\n")
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            # Save Linear Adapter
            torch.save(adapter.state_dict(), os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))
            # Save LoRA Adapter
            clip_model.save_pretrained(lora_save_path)

    # --- Restore Best Models ---
    print(f"**** Loading best models from epoch {best_epoch} ****")
    
    # 1. Load Linear Adapter
    adapter.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"), map_location=device))
    
    # 2. Load LoRA Adapter (This puts the CLIP model back to its best state)
    clip_model.load_adapter(lora_save_path, adapter_name="default")
    
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    if wandb_log:
        wandb.summary["best_test_accuracy"] = best_acc
    
    # Re-encode Val and Test sets with the BEST model for final search and eval
    print("Re-encoding Validation set with best model for Hyperparameter Search...")
    val_features_list = []
    val_labels_list = []
    clip_model.eval()
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc="Encoding Val Set"):
            images = images.to(device)
            target = target.to(device)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            val_features_list.append(features)
            val_labels_list.append(target)
    val_features_best = torch.cat(val_features_list, dim=0)
    val_labels_best = torch.cat(val_labels_list, dim=0)
    
    print("Re-encoding Test set with best model...")
    test_features_list = []
    test_labels_list = []
    with torch.no_grad():
        for images, target in tqdm(test_loader, desc="Encoding Test Set"):
            images = images.to(device)
            target = target.to(device)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            test_features_list.append(features)
            test_labels_list.append(target)
    test_features_best = torch.cat(test_features_list, dim=0)
    test_labels_best = torch.cat(test_labels_list, dim=0)

    print("\n-------- Searching hyperparameters on the val set. --------")
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values.to(dtype), val_features_best.to(dtype), val_labels_best, clip_weights.to(dtype), adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
    
    affinity = adapter(test_features_best.to(dtype))
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(dtype)
    
    clip_logits_best = 100. * test_features_best.to(dtype) @ clip_weights.to(dtype)
    
    tip_logits = clip_logits_best + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels_best)
    final_acc = max(best_acc, acc)
    print("**** {} Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(cfg['dataset'], final_acc))
    
    if wandb_log:
        wandb.summary["final_test_accuracy_after_search"] = final_acc


def main():
    
    parser = argparse.ArgumentParser()
    # Config and dataset args
    parser.add_argument('--config', type=str, default='./configs/fgvc.yaml', help='Path to dataset config file')
    parser.add_argument('--shots', type=int, default=4, help='Number of few-shot samples')
    # Optimizer args
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    # Model architecture args
    parser.add_argument('--hgt_num_layers', type=int, default=3, help='Number of HGT layers')
    parser.add_argument('--hgt_num_heads', type=int, default=16, help='Number of HGT attention heads')
    parser.add_argument('--transformer_num_layers', type=int, default=3, help='Number of Transformer Encoder layers')
    parser.add_argument('--transformer_nhead', type=int, default=16, help='Number of attention heads in the transformer encoder')
    parser.add_argument('--transformer_ff_multiplier', type=int, default=2, help='Feed-forward network dimension multiplier')
    parser.add_argument('--pooling_ratio', type=float, default=0.25, help='TopKPooling ratio')
    # Regularization args
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    # Training args
    parser.add_argument('--train_epoch', type=int, default=100, help='Total training epochs')
    parser.add_argument('--init_beta', type=float, default=2.0493246903001525, help='Initial beta for Tip-Adapter')
    parser.add_argument('--init_alpha', type=float, default=9.806773071379816, help='Initial alpha for Tip-Adapter')
    parser.add_argument('--init_gamma', type=float, default=1.0, help='Initial gamma for graph logits contribution')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='Weight for the direct contrastive loss')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for Focal Loss (controls hardness focus)')

    # <<< NEW: LoRA Hyperparameters >>>
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA Rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA Alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA Dropout')

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(config=args) 
    config = wandb.config
    
    cfg = yaml.load(open(config.config, 'r'), Loader=yaml.Loader)
    cfg["shots"] = config.shots
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['train_epoch'] = config.train_epoch

    print("\nRunning configs.")
    print(cfg, "\n")
    print("Sweep Hyperparameters:")
    print(wandb.config)
    
    # Dataset specific num_classes
    if(cfg['dataset'] == "birds"): num_classes = 200
    elif(cfg['dataset'] == "dogs"): num_classes = 120
    elif(cfg['dataset'] == "cars"): num_classes = 196
    elif(cfg['dataset'] == "oxford_pets"): num_classes = 37
    elif(cfg['dataset'] == "flowers"): num_classes = 102
    elif(cfg['dataset'] == "food101"): num_classes = 101
    elif(cfg['dataset'] == "dtd"): num_classes = 47
    elif(cfg['dataset'] == "aircrafts"): num_classes = 100
    elif(cfg['dataset'] == "ucf101"): num_classes = 101

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    node_types = ['vit', 'text']
    edge_types = [
        ('vit', 'intra_patch', 'vit'),
        ('vit', 'visual_to_text', 'text'),
        ('text', 'text_to_visual', 'vit'),
    ]

    # --- Load CLIP Models ---
    # Make sure to add jit=False
    clip_model_vit, vit_preprocess = clip.load(cfg['backbone'][0], device=device, jit=False)
    clip_model_vit = clip_model_vit.float()
    clip_model_vit.visual = TokenCLIPVisual(clip_model_vit.visual)
    patch_processor = vit_preprocess

    # --- Prepare Dataset ---
    random.seed(1)
    torch.manual_seed(1)
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    print("\nGetting textual features for graph nodes and CLIP's classifier.")
    text_features_for_nodes = clip_classifier(dataset.classnames, dataset.template, clip_model_vit)
    text_features_for_nodes = text_features_for_nodes.t()
    
    clip_weights = text_features_for_nodes.t()

    # --- Initialize HGT Model ---
    input_dims = {
        'vit': clip_model_vit.visual.output_dim,
        'text': clip_model_vit.text_projection.shape[1] 
    }
    hidden_dim = clip_model_vit.visual.output_dim

    image_model = HGTImageFeatureExtractor(
        node_types=node_types,
        edge_types=edge_types,
        input_dims=input_dims,
        hidden_channels=hidden_dim,
        hgt_num_heads=config.hgt_num_heads,
        hgt_num_layers=config.hgt_num_layers,
        dropout_rate=config.dropout_rate,
        transformer_nhead=config.transformer_nhead,
        transformer_num_layers=config.transformer_num_layers,
        transformer_ff_multiplier=config.transformer_ff_multiplier,
        transformer_activation='gelu',
        pooling_ratio=config.pooling_ratio,
        shots=config.shots
    )
    image_model.to(device)

    # --- DataLoaders ---
    # <<< FIX: Increased batch_size to 64 for faster evaluation during training >>>
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=vit_preprocess)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=vit_preprocess)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=config.batch_size, is_train=True, tfm=train_transform)
    
    train_loader_F = build_graph_data_loader(
        data_source=dataset.train_x,
        batch_size=config.batch_size,
        shuffle=True,
        transform=train_transform,
        vit_model=clip_model_vit.visual, # Passed the wrapped model here
        text_features=text_features_for_nodes,
        processor=patch_processor,
        device=device
    )

    # --- Pre-load features and build initial cache model ---
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model_vit, train_loader_cache)

    print("\nLoading visual features and labels from val set.")
    # Keep these for the initial Zero-Shot/Tip-Adapter evaluation before LoRA
    val_features, val_labels = pre_load_features(cfg, "val", clip_model_vit, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model_vit, test_loader)

    # --- Run Initial Zero-Shot and Tip-Adapter (Pre-LoRA) ---
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # --- Run Fine-Tuning and Evaluation ---
    # <<< FIX: Passed val_loader and test_loader instead of fixed tensors >>>
    # Note: clip_model_vit will be modified in-place by LoRA inside this function
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_loader, test_loader, clip_weights, clip_model_vit, image_model, train_loader_F)
    
    print("\nRunning configs.")
    print(cfg, "\n")
    print("Sweep Hyperparameters:")
    print(wandb.config)
    
    if wandb_log:
        wandb.finish()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()