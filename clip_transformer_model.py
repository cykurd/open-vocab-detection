"""Custom CLIP-Transformer model for zero-shot object detection following the proposed architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import math


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer between CLIP text embeddings and image features."""
    
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, text_embeddings, image_features):
        """
        Args:
            text_embeddings: [batch_size, text_seq_len, d_model] - CLIP text embeddings
            image_features: [batch_size, img_seq_len, d_model] - CLIP image features
        Returns:
            attended_features: [batch_size, img_seq_len, d_model]
        """
        # Cross-attention: image features attend to text embeddings
        # Q = image_features, K = V = text_embeddings
        attended, _ = self.attention(
            query=image_features,
            key=text_embeddings, 
            value=text_embeddings
        )
        
        # Residual connection and layer norm
        attended = self.norm1(image_features + attended)
        
        # Feed-forward network
        ffn_out = self.ffn(attended)
        output = self.norm2(attended + ffn_out)
        
        return output


class DetectionHead(nn.Module):
    """Detection head that outputs bounding boxes and confidence scores."""
    
    def __init__(self, d_model=512, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        
        # Object queries (learnable embeddings)
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Detection heads
        self.bbox_head = nn.Linear(d_model, 4)  # [x1, y1, x2, y2]
        self.confidence_head = nn.Linear(d_model, 1)  # objectness score
        
        # Initialize bbox head to predict boxes near center with small size
        # Bias for [x1, y1, x2, y2] ~ [0.3, 0.3, 0.7, 0.7] (small box at center)
        # sigmoid(x) ≈ 0.5 when x=0, so we want logit ≈ 0.5 -> logit ≈ -0.4 -> bias ≈ -0.4
        with torch.no_grad():
            self.bbox_head.bias.data = torch.tensor([-0.7, -0.7, -0.2, -0.2])  # Centers around [0.3, 0.3, 0.6, 0.6]
            self.bbox_head.weight.data *= 0.1  # Small weights to start
        
        # Initialize confidence head to low confidence initially
        with torch.no_grad():
            self.confidence_head.bias.data.fill_(-2.0)  # sigmoid(-2) ≈ 0.12 (low confidence)
        
        # Initialize object queries
        nn.init.normal_(self.query_embed.weight)
        
    def forward(self, cross_attended_features):
        """
        Args:
            cross_attended_features: [batch_size, img_seq_len, d_model]
        Returns:
            pred_boxes: [batch_size, num_queries, 4]
            pred_logits: [batch_size, num_queries, 1]
        """
        batch_size = cross_attended_features.size(0)
        
        # Get object queries
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention between object queries and cross-attended features
        # This allows queries to attend to relevant image regions
        attention_weights = torch.matmul(query_embeds, cross_attended_features.transpose(-2, -1))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Weighted combination of image features
        attended_queries = torch.matmul(attention_weights, cross_attended_features)
        
        # Generate predictions
        pred_boxes = self.bbox_head(attended_queries)
        pred_logits = self.confidence_head(attended_queries)
        
        # Apply sigmoid to get [0,1] normalized coordinates
        pred_boxes = torch.sigmoid(pred_boxes)
        
        # Ensure x1 < x2 and y1 < y2 by sorting coordinates
        # This prevents negative box areas and invalid IoU calculations
        pred_boxes = torch.stack([
            torch.min(pred_boxes[:, :, 0], pred_boxes[:, :, 2]),
            torch.min(pred_boxes[:, :, 1], pred_boxes[:, :, 3]),
            torch.max(pred_boxes[:, :, 0], pred_boxes[:, :, 2]),
            torch.max(pred_boxes[:, :, 1], pred_boxes[:, :, 3]),
        ], dim=-1)
        
        return pred_boxes, pred_logits


class CLIPTransformerDetector(nn.Module):
    """Complete CLIP-Transformer model for zero-shot object detection."""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16", 
                 d_model=512, nhead=8, num_layers=2, num_queries=100):
        super().__init__()
        
        # Load frozen CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP encoders
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Get CLIP dimensions
        self.clip_dim = self.clip_model.config.projection_dim  # 512 for ViT-B/16
        self.clip_vision_hidden = self.clip_model.vision_model.config.hidden_size  # 768 for ViT-B/16
        
        # Projection layers to match dimensions
        self.text_proj = nn.Linear(self.clip_dim, d_model)
        self.image_proj = nn.Linear(self.clip_vision_hidden, d_model)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Detection head
        self.detection_head = DetectionHead(d_model, num_queries)
        
    def freeze_for_phase1(self):
        """Freeze everything except cross-attention and detection head (Phase 1 training)."""
        for name, param in self.named_parameters():
            if any(x in name for x in ['cross_attention_layers', 'detection_head', 'text_proj', 'image_proj']):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def encode_text(self, text_queries):
        """Encode text queries using CLIP text encoder."""
        # Process text through CLIP
        text_inputs = self.clip_processor(text=text_queries, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(next(self.parameters()).device) for k, v in text_inputs.items()}
        
        # Get text embeddings
        text_outputs = self.clip_model.get_text_features(**text_inputs)
        text_embeddings = text_outputs.unsqueeze(1)  # [batch_size, 1, clip_dim]
        
        # Project to model dimension
        text_embeddings = self.text_proj(text_embeddings)  # [batch_size, 1, d_model]
        
        return text_embeddings
    
    def encode_image(self, images):
        """Encode images into per-patch features using CLIP's vision transformer.

        Returns [batch, num_patches, d_model].
        """
        # Process images through CLIP processor
        image_inputs = self.clip_processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(next(self.parameters()).device) for k, v in image_inputs.items()}

        # Use the vision transformer to get patch embeddings (exclude CLS token)
        vision_out = self.clip_model.vision_model(pixel_values=image_inputs["pixel_values"])  # BaseModelOutputWithPooling
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_dim]

        # Project to model dimension
        image_features = self.image_proj(patch_tokens)  # [B, num_patches, d_model]

        return image_features
    
    def forward(self, images, text_queries):
        """
        Forward pass through the complete pipeline.
        
        Args:
            images: List of PIL Images or tensor
            text_queries: List of text strings or single string
            
        Returns:
            outputs: Object with pred_boxes and logits attributes
        """
        # Ensure text_queries is a list
        if isinstance(text_queries, str):
            text_queries = [text_queries]
            
        # Encode text and images
        text_embeddings = self.encode_text(text_queries)
        image_features = self.encode_image(images)
        
        # Apply cross-attention layers
        cross_attended = image_features
        for layer in self.cross_attention_layers:
            cross_attended = layer(text_embeddings, cross_attended)
        
        # Generate detections
        pred_boxes, pred_logits = self.detection_head(cross_attended)
        
        # Create output object similar to GroundingDINO
        class Outputs:
            def __init__(self, pred_boxes, logits, region_embeddings=None):
                self.pred_boxes = pred_boxes
                self.logits = logits
                self.region_embeddings = region_embeddings  # For contrastive loss
                
        # Return region embeddings (the cross-attended features before detection head)
        # Shape: [batch_size, num_queries, d_model]
        return Outputs(pred_boxes, pred_logits, region_embeddings=cross_attended)


def load_clip_transformer_detector(model_name="openai/clip-vit-base-patch16", 
                                 d_model=512, nhead=8, num_layers=2, num_queries=100):
    """Load the custom CLIP-Transformer detector."""
    model = CLIPTransformerDetector(
        clip_model_name=model_name,
        d_model=d_model,
        nhead=nhead, 
        num_layers=num_layers,
        num_queries=num_queries
    )
    
    # Set up Phase 1 training (only cross-attention and detection head trainable)
    model.freeze_for_phase1()
    
    return model, model.clip_processor


if __name__ == "__main__":
    # Test the model
    model, processor = load_clip_transformer_detector()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Test forward pass
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Test forward pass
    with torch.no_grad():
        outputs = model([dummy_image], "road sky")
        print(f"Output shapes: boxes {outputs.pred_boxes.shape}, logits {outputs.logits.shape}")
