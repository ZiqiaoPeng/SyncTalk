import sys, os
import numpy as np
import torch


def make_conv(n_in, n_out, n_blocks, kernel=3, normalization=torch.nn.BatchNorm3d, activation=torch.nn.ReLU):
    blocks = []
    for i in range(n_blocks):                                                                                                                                                                                                                                                                                             
        in1 = n_in if i == 0 else n_out
        blocks.append(torch.nn.Sequential(
            torch.nn.Conv3d(in1, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        ))
    return torch.nn.Sequential(*blocks)


def make_conv_2d(n_in, n_out, n_blocks, kernel=3, normalization=torch.nn.BatchNorm2d, activation=torch.nn.ReLU):
    blocks = []
    for i in range(n_blocks):                                                                                                                                                                                                                                                                                             
        in1 = n_in if i == 0 else n_out
        blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in1, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        ))
    return torch.nn.Sequential(*blocks)


def make_downscale(n_in, n_out, kernel=4, normalization=torch.nn.BatchNorm3d, activation=torch.nn.ReLU):                                                                                                                                                                                                                              
    block = torch.nn.Sequential(
        torch.nn.Conv3d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block


def make_downscale_2d(n_in, n_out, kernel=4, normalization=torch.nn.BatchNorm2d, activation=torch.nn.ReLU):                                                                                                                                                                                                                              
    block = torch.nn.Sequential(
        torch.nn.Conv2d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block
    

def make_upscale(n_in, n_out, normalization=torch.nn.BatchNorm3d, activation=torch.nn.ReLU):
    block = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(n_in, n_out, kernel_size=6, stride=2, padding=2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block


def make_upscale_2d(n_in, n_out, kernel=4, normalization=torch.nn.BatchNorm2d, activation=torch.nn.ReLU):
    block = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block


class ResBlock(torch.nn.Module):
    def __init__(self, n_out, kernel=3, normalization=torch.nn.BatchNorm3d, activation=torch.nn.ReLU):
        super().__init__()
        self.block0 = torch.nn.Sequential(
            torch.nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        )
        
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
        )

        self.block2 = torch.nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)
        
        x = self.block2(x + x0)
        return x


class ResBlock2d(torch.nn.Module):
    def __init__(self, n_out, kernel=3, normalization=torch.nn.BatchNorm2d, activation=torch.nn.ReLU):
        super().__init__()
        self.block0 = torch.nn.Sequential(
            torch.nn.Conv2d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        )
        
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
        )

        self.block2 = torch.nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)
        
        x = self.block2(x + x0)
        return x


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x


def downscale_gt_flow(flow_gt, flow_mask, image_height, image_width):
    flow_gt_copy = flow_gt.clone()
    flow_mask_copy = flow_mask.clone()

    flow_gt_copy = flow_gt_copy / 20.0
    flow_mask_copy = flow_mask_copy.float()

    assert image_height % 64 == 0 and image_width % 64 == 0

    flow_gt2 = torch.nn.functional.interpolate(input=flow_gt_copy, size=(image_height//4, image_width//4), mode='nearest')
    flow_mask2 = torch.nn.functional.interpolate(input=flow_mask_copy, size=(image_height//4, image_width//4), mode='nearest').bool()
    
    flow_gt3 = torch.nn.functional.interpolate(input=flow_gt_copy, size=(image_height//8, image_width//8), mode='nearest')
    flow_mask3 = torch.nn.functional.interpolate(input=flow_mask_copy, size=(image_height//8, image_width//8), mode='nearest').bool()
    
    flow_gt4 = torch.nn.functional.interpolate(input=flow_gt_copy, size=(image_height//16, image_width//16), mode='nearest')
    flow_mask4 = torch.nn.functional.interpolate(input=flow_mask_copy, size=(image_height//16, image_width//16), mode='nearest').bool()
    
    flow_gt5 = torch.nn.functional.interpolate(input=flow_gt_copy, size=(image_height//32, image_width//32), mode='nearest')
    flow_mask5 = torch.nn.functional.interpolate(input=flow_mask_copy, size=(image_height//32, image_width//32), mode='nearest').bool()

    flow_gt6 = torch.nn.functional.interpolate(input=flow_gt_copy, size=(image_height//64, image_width//64), mode='nearest')
    flow_mask6 = torch.nn.functional.interpolate(input=flow_mask_copy, size=(image_height//64, image_width//64), mode='nearest').bool()

    return [flow_gt2, flow_gt3, flow_gt4, flow_gt5, flow_gt6], [flow_mask2, flow_mask3, flow_mask4, flow_mask5, flow_mask6]


def compute_baseline_mask_gt(
    xy_coords_warped, 
    target_matches, valid_target_matches,
    source_points, valid_source_points,
    scene_flow_gt, scene_flow_mask, target_boundary_mask,
    max_pos_flowed_source_to_target_dist, min_neg_flowed_source_to_target_dist
):
    # Scene flow mask
    scene_flow_mask_0 = scene_flow_mask[:, 0].type(torch.bool)

    # Boundary correspondences mask
    # We use the nearest neighbor interpolation, since the boundary computations
    # already marks any of 4 pixels as boundary.
    target_nonboundary_mask = (~target_boundary_mask).type(torch.float32)
    target_matches_nonboundary_mask = torch.nn.functional.grid_sample(target_nonboundary_mask, xy_coords_warped, padding_mode='zeros', mode='nearest', align_corners=False)
    target_matches_nonboundary_mask = target_matches_nonboundary_mask[:, 0, :, :] >= 0.999

    # Compute groundtruth mask (oracle)
    flowed_source_points = source_points + scene_flow_gt
    dist = torch.norm(flowed_source_points - target_matches, p=2, dim=1)

    # Combine all masks
    # We mark a correspondence as positive if;
    # - it is close enough to groundtruth flow 
    # AND
    # - there exists groundtruth flow 
    # AND
    # - the target match is valid
    # AND
    # - the source point is valid
    # AND
    # - the target match is not on the boundary
    mask_pos_gt = (dist <= max_pos_flowed_source_to_target_dist) & scene_flow_mask_0 & valid_target_matches & valid_source_points & target_matches_nonboundary_mask

    # We mark a correspondence as negative if;
    # - there exists groundtruth flow AND it is far away enough from the groundtruth flow AND source/target points are valid
    # OR
    # - the target match is on the boundary AND there exists groundtruth flow AND source/target points are valid
    mask_neg_gt = ((dist > min_neg_flowed_source_to_target_dist) & scene_flow_mask_0 & valid_source_points & valid_target_matches) \
            | (~target_matches_nonboundary_mask & scene_flow_mask_0 & valid_source_points & valid_target_matches)

    # What remains is left undecided (masked out at loss).
    # For groundtruth mask we set it to zero.
    valid_mask_pixels = mask_pos_gt | mask_neg_gt
    mask_gt = mask_pos_gt

    mask_gt = mask_gt.type(torch.float32)
    
    return mask_gt, valid_mask_pixels


def compute_deformed_points_gt(
    source_points, scene_flow_gt, 
    valid_solve, valid_correspondences, 
    deformed_points_idxs, deformed_points_subsampled
):
    batch_size = source_points.shape[0]
    max_warped_points = deformed_points_idxs.shape[1]

    deformed_points_gt = torch.zeros((batch_size, max_warped_points, 3), dtype=source_points.dtype, device=source_points.device) 
    deformed_points_mask = torch.zeros((batch_size, max_warped_points, 3), dtype=source_points.dtype, device=source_points.device) 

    for i in range(batch_size):
        if valid_solve[i]:
            valid_correspondences_idxs = torch.where(valid_correspondences[i])

            # Compute deformed point groundtruth.
            deformed_points_i_gt = source_points[i] + scene_flow_gt[i]
            deformed_points_i_gt = deformed_points_i_gt.permute(1, 2, 0)
            deformed_points_i_gt = deformed_points_i_gt[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(-1, 3, 1)

            # Filter out points randomly, if too many are still left.
            if deformed_points_subsampled[i]:
                sampled_idxs_i = deformed_points_idxs[i]
                deformed_points_i_gt  = deformed_points_i_gt[sampled_idxs_i]

            num_points = deformed_points_i_gt.shape[0]

            # Store the results.
            deformed_points_gt[i, :num_points, :] = deformed_points_i_gt.view(1, num_points, 3)
            deformed_points_mask[i, :num_points, :] = 1

    return deformed_points_gt, deformed_points_mask
