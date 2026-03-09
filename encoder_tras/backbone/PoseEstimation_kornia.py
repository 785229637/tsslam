import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia as K
import numpy as np
from torch.hub import load_state_dict_from_url
import os


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=14, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim **-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTS14(nn.Module):
    def __init__(self, img_size=256, patch_size=14, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class ViTS14FeatureExtractor(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        self.patch_size = 14
        self.embed_dim = 384
        self.img_size = 256
        self.vit = ViTS14(img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.embed_dim, depth=12, num_heads=6)
        self._load_vits_weights(weights_path)
        self.patch_embed = self.vit.patch_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

    def _load_vits_weights(self, weights_path):
        vits_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
        default_cache_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
        
        if weights_path is not None and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
        elif os.path.exists(default_cache_path):
            state_dict = torch.load(default_cache_path, map_location="cpu")
        else:
            try:
                print("正在下载ViT-S/14权重...")
                state_dict = load_state_dict_from_url(vits_url, progress=True)
            except Exception as e:
                raise RuntimeError(f"权重下载失败: {e}\n请手动下载: {vits_url}")
        
        model_state = self.vit.state_dict()
        adjusted_state = {}
        
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                k = k[len('backbone.'):]
            

            if k == 'pos_embedding':
                if v.shape[1] == self.vit.num_patches + 1:
                    v = v[:, 1:, :]
                
                pretrained_grid_size = int(v.shape[1]** 0.5)
                our_grid_size = self.vit.patch_embed.grid_size[0]
                
                v_2d = v.reshape(1, pretrained_grid_size, pretrained_grid_size, -1).permute(0, 3, 1, 2)
                v_resized = F.interpolate(v_2d, size=(our_grid_size, our_grid_size), mode='bilinear', align_corners=False)
                v = v_resized.permute(0, 2, 3, 1).reshape(1, our_grid_size * our_grid_size, -1)
                k = 'pos_embed'
            
            if k in model_state and v.shape == model_state[k].shape:
                adjusted_state[k] = v
        
        self.vit.load_state_dict(adjusted_state, strict=False)

    def forward(self, x):
        x = self.vit(x)  # [B, num_patches, 384]
        return {"x_norm_patchtokens": x}


class BFMatcher(nn.Module):
    def __init__(self, ratio_test=0.85):
        super().__init__()
        self.ratio_test = ratio_test

    def forward(self, desc1, desc2):
        B, N1, D = desc1.shape
        N2 = desc2.shape[1]
        
        similarity = torch.bmm(desc1, desc2.transpose(1, 2))
        top2_vals, top2_idx = torch.topk(similarity, k=2, dim=2)
        
        ratio = top2_vals[..., 0] / (top2_vals[..., 1] + 1e-8)
        mask = ratio < self.ratio_test
        
        matches = []
        for b in range(B):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                matches.append(torch.empty((0, 2), dtype=torch.long, device=desc1.device))
                continue
            
            src_idx = valid_idx
            dst_idx = top2_idx[b, valid_idx, 0]
            batch_matches = torch.stack([src_idx, dst_idx], dim=1)
            matches.append(batch_matches)
        
        max_matches = max([m.shape[0] for m in matches]) if matches else 0
        for i in range(len(matches)):
            if matches[i].shape[0] < max_matches:
                pad = torch.zeros((max_matches - matches[i].shape[0], 2), 
                                 dtype=torch.long, device=desc1.device)
                matches[i] = torch.cat([matches[i], pad], dim=0)
        
        return torch.stack(matches, dim=0)


def get_match_points(kps, match_indices):
    B, M = match_indices.shape
    matched_kps = []
    
    for b in range(B):
        indices = match_indices[b]
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        
        if valid_indices.numel() == 0:
            matched = torch.empty((0, 2), device=kps.device)
        else:
            matched = kps[b].index_select(0, valid_indices)
            
            if valid_mask.sum() < M:
                pad_size = M - valid_mask.sum()
                pad = torch.zeros((pad_size, 2), device=kps.device)
                matched = torch.cat([matched, pad], dim=0)
        
        matched_kps.append(matched)
    
    return torch.stack(matched_kps, dim=0)


def find_fundamental(src_pts, dst_pts, method='ransac', prob=0.999, threshold=0.5, max_iter=2000):
    B, N, _ = src_pts.shape
    device = src_pts.device
    F_batch = []
    mask_batch = []
    
    for b in range(B):

        src = src_pts[b].cpu().numpy()
        dst = dst_pts[b].cpu().numpy()
        

        if N < 8:
            F = torch.eye(3, device=device) * 1e-8
            mask = torch.zeros(N, device=device, dtype=torch.bool)
            F_batch.append(F)
            mask_batch.append(mask)
            continue
        

        best_inliers = 0
        best_F = None
        best_mask = None
        iter = 0
        

        src_hom = np.hstack([src, np.ones((N, 1))])  # [N, 3]
        dst_hom = np.hstack([dst, np.ones((N, 1))])  # [N, 3]
        
        while iter < max_iter:

            idx = np.random.choice(N, 8, replace=False)
            src_sample = src[idx]
            dst_sample = dst[idx]
            

            F = _compute_fundamental_matrix(src_sample, dst_sample)
            if F is None:
                iter += 1
                continue
            

            distances = _sampson_distance(F, src_hom, dst_hom)
            

            mask = distances < threshold
            inlier_count = np.sum(mask)
            

            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_F = F
                best_mask = mask
                

                if best_inliers > 0:
                    inlier_ratio = best_inliers / N
                    if inlier_ratio > 0:

                        p_no_outlier = inlier_ratio **8
                        if p_no_outlier > 0:
                            new_iter = np.log(1 - prob) / np.log(1 - p_no_outlier)
                            max_iter = min(max_iter, int(new_iter) + iter)
            
            iter += 1
        

        if best_F is None:
            best_F = np.zeros((3, 3))
            best_mask = np.zeros(N, dtype=bool)
        

        F_batch.append(torch.tensor(best_F, device=device, dtype=torch.float32))
        mask_batch.append(torch.tensor(best_mask, device=device, dtype=torch.bool))
    
    return torch.stack(F_batch, dim=0), torch.stack(mask_batch, dim=0)

def _compute_fundamental_matrix(src, dst):
    n = src.shape[0]
    if n < 8:
        return None
    

    A = []
    for i in range(n):
        x1, y1 = src[i]
        x2, y2 = dst[i]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    A = np.array(A)
    

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    

    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0
    F = Uf @ np.diag(Sf) @ Vtf
    

    return F / F[2, 2] if F[2, 2] != 0 else F

def _sampson_distance(F, src_hom, dst_hom):
    # Fx1
    Fx1 = F @ src_hom.T  # [3, N]
    # x2^T F
    x2TF = dst_hom @ F  # [N, 3]
    

    numerator = (dst_hom @ F @ src_hom.T.diagonal())** 2  # [N]
    

    denominator = Fx1[0, :]**2 + Fx1[1, :]** 2 + x2TF[:, 0]**2 + x2TF[:, 1]** 2  # [N]
    

    denominator = np.maximum(denominator, 1e-8)
    
    return np.sqrt(numerator / denominator)  # [N]

def triangulate_points(P1, P2, pts1, pts2):
    N = pts1.shape[0]
    points3D = []
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A = [
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ]
        A = np.stack(A, axis=0)
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        points3D.append(X[:3])
    
    return np.array(points3D)

class PoseEstimationHead(nn.Module):

    def __init__(self, dino_weights_path=None, camk=None):
        super().__init__()
        self.feature_extractor = ViTS14FeatureExtractor(weights_path=dino_weights_path)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        

        self.feature_adjuster = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=True)
        )
        
        self.matcher = BFMatcher(ratio_test=0.85)
        

        if camk is not None:

            self.K = nn.Parameter(
                camk[:, :, :3, :3].squeeze(0),
                requires_grad=False
            )
        else:

            self.K = nn.Parameter(
                torch.tensor([[200.0, 0.0, 128.0],
                             [0.0, 200.0, 128.0],
                             [0.0, 0.0, 1.0]]),
                requires_grad=False
            )

    def extract_features(self, x):

        B, C, H, W = x.shape
        features = self.feature_extractor(x)
        patch_feats = features["x_norm_patchtokens"]
        patch_size = self.feature_extractor.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        feat_map = patch_feats.permute(0, 2, 1).view(B, 384, h_patches, w_patches)
        
        kps = []
        for i in range(h_patches):
            for j in range(w_patches):
                x_coord = (j + 0.5) * patch_size
                y_coord = (i + 0.5) * patch_size
                kps.append([x_coord, y_coord])
        kps = torch.tensor(kps, device=x.device).repeat(B, 1, 1)
        
        descs = F.normalize(patch_feats, dim=2)
        return feat_map, kps, descs

    def forward(self, images, camk=None):

        if camk is not None:
            current_K = camk[:, :, :3, :3].squeeze(0).to(images.device)
        else:
            current_K = self.K.to(images.device)
        
        img1 = images[:, 0]  # [B, 3, H, W]
        img2 = images[:, 1]  # [B, 3, H, W]
        
        transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img1 = transform(img1)
        img2 = transform(img2)
        

        feat_map1, kps1, descs1 = self.extract_features(img1)
        feat_map2, kps2, descs2 = self.extract_features(img2)
        

        adj_feat1 = self.feature_adjuster(feat_map1)[..., 4:68, 4:68]
        adj_feat2 = self.feature_adjuster(feat_map2)[..., 4:68, 4:68]
        merged_feats = torch.cat([adj_feat1, adj_feat2], dim=0)
        features_next = [merged_feats]
        

        matches = self.matcher(descs1, descs2)
        src_pts = get_match_points(kps1, matches[:, :, 0])  # [B, M, 2]
        dst_pts = get_match_points(kps2, matches[:, :, 1])
        

        F, mask = find_fundamental(src_pts, dst_pts, method='ransac', prob=0.999, threshold=0.5)
        

        B, M, _ = src_pts.shape
        inlier_src_pts = []
        inlier_dst_pts = []
        for b in range(B):
            valid_mask = mask[b]
            inlier_src = src_pts[b][valid_mask]
            inlier_dst = dst_pts[b][valid_mask]
            inlier_src_pts.append(inlier_src.cpu().numpy())
            inlier_dst_pts.append(inlier_dst.cpu().numpy())
        

        K1 = current_K[0].unsqueeze(0).repeat(B, 1, 1)
        K2 = current_K[1].unsqueeze(0).repeat(B, 1, 1)
        E = K.geometry.epipolar.essential_from_fundamental(F, K1, K2)
        

        R1, R2, t = K.geometry.epipolar.decompose_essential_matrix(E)

        R1 = R1.squeeze(1) if R1.dim() == 4 else R1
        R2 = R2.squeeze(1) if R2.dim() == 4 else R2
        t = t.squeeze(1) if t.dim() == 3 else t
        

        best_num_positive = -1
        best_R = R1
        best_t = t
        
        for b in range(B):
            K1_np = K1[b].cpu().numpy()
            K2_np = K2[b].cpu().numpy()
            P1 = K1_np @ np.hstack([np.eye(3), np.zeros((3, 1))])  # [3,4]
            

            for R in [R1[b].cpu().numpy(), R2[b].cpu().numpy()]:
                for t_vec in [t[b].cpu().numpy(), -t[b].cpu().numpy()]:
                    P2 = K2_np @ np.hstack([R, t_vec.reshape(3, 1)])  # [3,4]
                    
                    if len(inlier_src_pts[b]) < 5:
                        continue
                    points3D = triangulate_points(
                        P1, P2, 
                        inlier_src_pts[b], 
                        inlier_dst_pts[b]
                    )
                    

                    z1 = points3D[:, 2]
                    positive_z1 = np.sum(z1 > 1e-4)
                    
                    X2 = R @ points3D.T + t_vec.reshape(3, 1)
                    z2 = X2[2, :]
                    positive_z2 = np.sum(z2 > 1e-4)
                    
                    total_positive = positive_z1 + positive_z2
                    
                    if total_positive > best_num_positive:
                        best_num_positive = total_positive
                        best_R = torch.tensor(R, device=R1.device, dtype=R1.dtype).unsqueeze(0)
                        best_t = torch.tensor(t_vec, device=t.device, dtype=t.dtype).unsqueeze(0)
        

        quat = K.geometry.conversions.rotation_matrix_to_quaternion(best_R)
        rotations = quat.unsqueeze(1)
        translations = best_t.unsqueeze(1)
        
        return rotations, translations, features_next
