import os
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch_geometric.data import Batch
import random
import os.path as osp
from collections import defaultdict
import json
import torch.nn as nn

# Assuming this function is in a file named graph_builder.py
from graph_builder import build_fully_connected_hetero_graph



# --- 1. CLIP Wrapper to force Token Output ---
class TokenCLIPVisual(nn.Module):
    """
    Wraps the visual part of CLIP to return the full sequence of tokens 
    (CLS + Patches) instead of just the pooled CLS embedding.
    """
    def __init__(self, original_visual):
        super().__init__()
        self.original = original_visual
        
        # --- FIX: Expose attributes required by the parent CLIP class ---
        # CLIP's .dtype property checks self.visual.conv1.weight.dtype
        self.conv1 = original_visual.conv1 
        
        # Expose other common attributes just in case
        self.class_embedding = original_visual.class_embedding
        self.positional_embedding = original_visual.positional_embedding
        self.ln_pre = original_visual.ln_pre
        self.transformer = original_visual.transformer
        self.ln_post = original_visual.ln_post
        self.proj = original_visual.proj
        
        self.input_resolution = original_visual.input_resolution
        self.output_dim = original_visual.output_dim
        
    def forward(self, x):
        # Access the internal methods of the original VisualTransformer
        model = self.original
        
        # 1. Convolution to get patches
        x = model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # 2. Add CLS token and Positional Embeddings
        x = torch.cat([model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + model.positional_embedding.to(x.dtype)
        x = model.ln_pre(x)

        # 3. Transformer Layers
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 4. Final Norm
        x = model.ln_post(x)

        # 5. Project if necessary
        if model.proj is not None:
            x = x @ model.proj

        # Output shape: [Batch, 197, Output_Dim]
        return x

    
# --- 2. Pre-compute Grid Topology ---
def create_grid_edge_index(grid_size=14):
    """
    Creates edges for a grid where every node is connected to its 8 neighbors.
    Node 0 is CLS. Nodes 1..196 are the 14x14 grid.
    """
    src_nodes = []
    dst_nodes = []
    
    # Offsets for 8 neighbors (King's moves)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    for y in range(grid_size):
        for x in range(grid_size):
            # Map grid (y,x) to node index (1 to 196)
            current_node = 1 + y * grid_size + x
            
            # A. Connect to Spatial Neighbors
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    neighbor_node = 1 + ny * grid_size + nx
                    src_nodes.append(current_node)
                    dst_nodes.append(neighbor_node)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index




    

# --- I/O and Helper Functions ---

def read_json(fpath: str) -> dict:
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj: dict, fpath: str):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def read_image(path: str) -> Image.Image:
    """Read image from path using PIL.Image, retrying on IOError."""
    if not os.path.exists(path):
        raise IOError(f'No file exists at {path}')
    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(f'Cannot read image from {path}, retrying...')

def listdir_nohidden(path: str, sort: bool = False) -> list[str]:
    """List non-hidden items in a directory."""
    items = [f for f in os.listdir(path) if not f.startswith('.')]
    if sort:
        items.sort()
    return items

# --- Data Structures ---

class Datum:
    """
    Data instance which defines the basic attributes for an image, including
    classname and domain for compatibility with DatasetBase.
    """
    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

class DatasetBase:
    """
    A unified dataset class for managing train/val/test splits, class names,
    and other dataset-level information.
    """
    dataset_dir = ''
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
    
    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')
        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)
            output.append(dataset)
        if len(output) == 1:
            return output[0]
        return output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

# --- Image Processing ---

# def fixed_grid_crops(img: Image.Image, crop_size: int = 224, nodes: int = 17) -> list[Image.Image]:
#     """
#     Generate a deterministic set of crops from a 336x336 image.
#     The default behavior produces 18 crops in total.
#     """
#     resize = T.Resize(size=(336, 336), interpolation=T.InterpolationMode.BICUBIC)
#     resize224 = T.Resize(size=(224, 224), interpolation=T.InterpolationMode.BICUBIC)
#     img = resize(img)
#     w, h = img.size
#     assert w == 336 and h == 336, f"Expected image of size 336x336, but got {w}x{h}"
    
#     crops = []
    
#     crops.append(resize224(img))
    
#     corners = [(0, 0), (w - crop_size, 0), (0, h - crop_size), (w - crop_size, h - crop_size)]
#     for x, y in corners:
#         crops.append(TF.crop(img, top=y, left=x, height=crop_size, width=crop_size))

#     crops.append(T.CenterCrop(crop_size)(img))
    
#     stride = crop_size // 2
#     shifts = [(stride, 0), (0, stride), (2 * stride, stride), (stride, 2 * stride)]
#     for x, y in shifts:
#         shifted_crop = TF.crop(img, top=y, left=x, height=stride, width=stride)
#         crops.append(resize224(shifted_crop))

#     quadrant_size = 168
#     quadrants = [(0, 0), (w - quadrant_size, 0), (0, h - quadrant_size), (w - quadrant_size, h - quadrant_size)]
#     for x, y in quadrants:
#         q_crop = TF.crop(img, top=y, left=x, height=quadrant_size, width=quadrant_size)
#         crops.append(resize224(q_crop))

#     horizontal_halves = [(0, 0), (0, h // 2)]
#     for x, y in horizontal_halves:
#         h_half_crop = TF.crop(img, top=y, left=x, height=168, width=336)
#         crops.append(resize224(h_half_crop))

#     vertical_halves = [(0, 0), (w // 2, 0)]
#     for x, y in vertical_halves:
#         v_half_crop = TF.crop(img, top=y, left=x, height=336, width=168)
#         crops.append(resize224(v_half_crop))
        
#     return crops


def fixed_grid_crops(img: Image.Image, crop_size: int = 224, nodes=17) -> list[Image.Image]:
    """
    Generates a multi-scale grid of patches from a single image.

    The process creates:
    - 1 full-image view
    - 9 patches from a 3x3 grid
    - 4 patches from a 2x2 grid
    - 2 patches from a vertical split
    - 2 patches from a horizontal split
    
    Total patches = 1 + 9 + 4 + 2 + 2 = 18.
    """
    crops = []
    
    # Standard resizer for crops that aren't already the target size
    resize_to_crop = T.Resize(size=(crop_size, crop_size), interpolation=T.InterpolationMode.BICUBIC)

    # --- 1. The original full image, resized ---
    # This is the global view.
    crops.append(resize_to_crop(img))

    # --- 2. Generate 3x3 grid (9 patches) ---
    # Resize the image to a size perfectly divisible by 3 (e.g., 672x672)
    grid_3x3_size = crop_size * 3
    img_3x3 = T.Resize(size=(grid_3x3_size, grid_3x3_size), interpolation=T.InterpolationMode.BICUBIC)(img)
    patch_size_3x3 = grid_3x3_size // 3  # This will be `crop_size`

    for i in range(3):
        for j in range(3):
            top = i * patch_size_3x3
            left = j * patch_size_3x3
            patch = TF.crop(img_3x3, top, left, patch_size_3x3, patch_size_3x3)
            crops.append(patch)

    # --- 3. Generate 2x2 grid (4 patches) ---
    # Resize the image to a size perfectly divisible by 2 (e.g., 448x448)
    grid_2x2_size = crop_size * 2
    img_2x2 = T.Resize(size=(grid_2x2_size, grid_2x2_size), interpolation=T.InterpolationMode.BICUBIC)(img)
    patch_size_2x2 = grid_2x2_size // 2 # This will be `crop_size`

    for i in range(2):
        for j in range(2):
            top = i * patch_size_2x2
            left = j * patch_size_2x2
            patch = TF.crop(img_2x2, top, left, patch_size_2x2, patch_size_2x2)
            crops.append(patch)
            
    # --- 4. Generate vertical halves (2 patches) ---
    # Use the 2x2 resized image (448x448) for this split
    w, h = img_2x2.size
    
    # Crop left half (448x224) and resize to 224x224
    left_half = TF.crop(img_2x2, top=0, left=0, height=h, width=w // 2)
    crops.append(resize_to_crop(left_half))
    
    # Crop right half (448x224) and resize to 224x224
    right_half = TF.crop(img_2x2, top=0, left=w // 2, height=h, width=w // 2)
    crops.append(resize_to_crop(right_half))

    # --- 5. Generate horizontal halves (2 patches) ---
    # Use the same 2x2 resized image (448x448)
    
    # Crop top half (224x448) and resize to 224x224
    top_half = TF.crop(img_2x2, top=0, left=0, height=h // 2, width=w)
    crops.append(resize_to_crop(top_half))
    
    # Crop bottom half (224x448) and resize to 224x224
    bottom_half = TF.crop(img_2x2, top=h // 2, left=0, height=h // 2, width=w)
    crops.append(resize_to_crop(bottom_half))

    # Sanity check to ensure we have the correct number of patches
    assert len(crops) == 18, f"Expected 18 patches, but generated {len(crops)}"
    
    return crops

# --- Standard DataLoader (for val/test) ---

class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, transform=None, is_train=False):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = read_image(item.impath)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, item.label

def build_data_loader(data_source, batch_size, is_train=False, tfm=None, shuffle=False):
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    return data_loader

# --- Graph-based DataLoader (for training) ---

# --- 3. Updated Dataset (No more crops) ---
class ImagePatchDataset(TorchDataset):
    def __init__(self, data_source, transform=None, patch_processor=None):
        self.data_source = data_source
        self.transform = transform
        # patch_processor is kept for compatibility but not used for cropping anymore

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        # Read and transform the full image
        try:
            image = Image.open(item.impath).convert('RGB')
        except IOError:
            print(f"Error reading {item.impath}")
            # Return a dummy black image in case of error to prevent crash
            image = Image.new('RGB', (224, 224))
            
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        return {
            'image': image_tensor,
            'label': torch.tensor(item.label, dtype=torch.long),
            'image_id': item.impath
        }


# --- 4. Updated Collate Function ---
class GraphCollate:
    def __init__(self, vit_feature_extractor, text_features, device):
        self.vit_model = vit_feature_extractor.to(device).eval()
        self.text_features = text_features.to(device)
        self.device = device
        
        # Pre-compute edges once. 
        # ViT-B/16 or B/32 on 224x224 yields 14x14 or 7x7 grid.
        # Assuming ViT-B/16 (14x14 = 196 patches).
        # If input_resolution=224 and patch_size=16 -> 14x14.
        self.grid_size = 14 
        self.visual_edge_index = create_grid_edge_index(self.grid_size).to(device)

    def __call__(self, batch: list[dict]):
        # Stack images: [Batch_Size, 3, 224, 224]
        images = torch.stack([item['image'] for item in batch]).to(self.device)
        labels = torch.stack([item['label'] for item in batch]).to(self.device)
        
        # 1. Run ViT once on the batch of images
        # Expect output: [Batch_Size, 197, Hidden_Dim]
        with torch.no_grad():
            batch_vit_tokens = self.vit_model(images) 

        graph_list = []
        B, N, D = batch_vit_tokens.shape 
        
        # 2. Construct HeteroGraphs
        # We reuse the same text_features for every graph
        num_text_nodes = self.text_features.shape[0]
        
        # Pre-create edges for Visual <-> Text
        # Connect CLS (Node 0) to All Text Nodes
        cls_idx = torch.zeros(num_text_nodes, dtype=torch.long, device=self.device)
        text_idx = torch.arange(num_text_nodes, dtype=torch.long, device=self.device)
        v2t_edge_index = torch.stack([cls_idx, text_idx], dim=0)
        t2v_edge_index = torch.stack([text_idx, cls_idx], dim=0)

        for i in range(B):
            vit_f = batch_vit_tokens[i] # [197, D]
            
            data = HeteroData()
            
            # Node Features
            data['vit'].x = vit_f
            data['text'].x = self.text_features
            
            # Edges
            data['vit', 'intra_patch', 'vit'].edge_index = self.visual_edge_index
            data['vit', 'visual_to_text', 'text'].edge_index = v2t_edge_index
            data['text', 'text_to_visual', 'vit'].edge_index = t2v_edge_index
            
            graph_list.append(data)
        
        batched_graphs = Batch.from_data_list(graph_list)
        
        return batched_graphs, images, labels
    
    
    
    
def build_graph_data_loader(
    data_source, batch_size, shuffle, transform, 
    vit_model, text_features, processor, device, num_workers=4
):
    dataset = ImagePatchDataset(data_source, transform=transform, patch_processor=processor)
    
    collate_fn = GraphCollate(
        vit_feature_extractor=vit_model,
        text_features=text_features,
        device=device
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return data_loader