# %%
FILE_1 = "Data/case6_gre1.nrrd"
FILE_2 = "Data/case6_gre2.nrrd"

# %%
import itk
import vtk
import matplotlib.pyplot as plt
import numpy as np

# %%
PixelType = itk.F

image1 = itk.imread(FILE_1, PixelType)
image2 = itk.imread(FILE_2, PixelType)

# %%
# Information about the images
print(f"Image 1: {image1}")
print(f"Image 2: {image2}")

# %% [markdown]
# # 2. Recalage

# %%
def extract_slice(volume, slice_index, axis=0):
    volume_array = itk.array_view_from_image(volume)
    
    if axis == 0:  # axial
        slice_array = volume_array[slice_index, :, :]
    elif axis == 1:  # coronal
        slice_array = volume_array[:, slice_index, :]
    elif axis == 2:  # sagittal
        slice_array = volume_array[:, :, slice_index]
    
    slice_image = itk.image_from_array(slice_array.astype(np.float32))
    
    spacing_3d = volume.GetSpacing()
    if axis == 0:
        spacing_2d = [spacing_3d[1], spacing_3d[2]]
    elif axis == 1:
        spacing_2d = [spacing_3d[0], spacing_3d[2]]
    else:  # axis == 2
        spacing_2d = [spacing_3d[0], spacing_3d[1]]
    
    slice_image.SetSpacing(spacing_2d)
    
    return slice_image

# %%
def register_2d_slice(fixed_slice, moving_slice, transform_type='translation'):
    dimension = 2
    FixedImageType = type(fixed_slice)
    MovingImageType = type(moving_slice)
    

    if transform_type == 'translation':
        TransformType = itk.TranslationTransform[itk.D, dimension]
        learning_rate = 4.0
        iterations = 100
    else:  # rigid
        TransformType = itk.Euler2DTransform[itk.D]
        learning_rate = 0.5
        iterations = 200
    
    initial_transform = TransformType.New()
    initial_transform.SetIdentity()
    
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    optimizer.SetLearningRate(learning_rate)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(iterations)
    optimizer.SetRelaxationFactor(0.5)
    
    metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()
    
    registration = itk.ImageRegistrationMethodv4.New(
        FixedImage=fixed_slice,
        MovingImage=moving_slice,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=initial_transform
    )
    
    registration.SetNumberOfLevels(1)
    
    try:
        registration.Update()
        transform = registration.GetTransform()
        return transform, True
    except:
        return initial_transform, False

# %%
def apply_transform_to_slice(moving_slice, fixed_slice, transform):
    resampler = itk.ResampleImageFilter.New(
        Input=moving_slice,
        Transform=transform,
        UseReferenceImage=True,
        ReferenceImage=fixed_slice
    )
    resampler.SetDefaultPixelValue(0)
    resampler.Update()
    return resampler.GetOutput()

# %%
def insert_slice_into_volume(volume, slice_2d, slice_index, axis=0):
    volume_array = itk.array_view_from_image(volume)
    slice_array = itk.array_view_from_image(slice_2d)
    
    # Insert the slice based on axis
    if axis == 0:  # axial
        volume_array[slice_index, :, :] = slice_array
    elif axis == 1:  # coronal
        volume_array[:, slice_index, :] = slice_array
    else:  # sagittal
        volume_array[:, :, slice_index] = slice_array
    
    return volume

# %%
def register_volume_slice_by_slice(fixed_volume, moving_volume, 
                                   transform_type='translation', 
                                   axis=0,
                                   show_progress=True):
    size = fixed_volume.GetLargestPossibleRegion().GetSize()
    if axis == 0:
        num_slices = size[0]
    elif axis == 1:
        num_slices = size[1]
    else:
        num_slices = size[2]

    OutputImageType = type(fixed_volume)
    output_volume = OutputImageType.New()
    output_volume.SetRegions(fixed_volume.GetLargestPossibleRegion())
    output_volume.SetSpacing(fixed_volume.GetSpacing())
    output_volume.SetOrigin(fixed_volume.GetOrigin())
    output_volume.SetDirection(fixed_volume.GetDirection())
    output_volume.Allocate()
    output_volume.FillBuffer(0.0)
    
    translations_x = []
    translations_y = []
    rotations = []
    
    print(f"Registering {num_slices} slices...")
    
    for slice_idx in range(num_slices):
        if show_progress and slice_idx % 20 == 0:
            print(f"Processing slice {slice_idx}/{num_slices}")
        
        fixed_slice = extract_slice(fixed_volume, slice_idx, axis)
        moving_slice = extract_slice(moving_volume, slice_idx, axis)
        
        transform, success = register_2d_slice(fixed_slice, moving_slice, transform_type)
        
        if success:
            params = transform.GetParameters()
            if transform_type == 'translation':
                translations_x.append(params[0])
                translations_y.append(params[1])
            else:  # rigid
                translations_x.append(params[1])
                translations_y.append(params[2])
                rotations.append(params[0])
            
            registered_slice = apply_transform_to_slice(moving_slice, fixed_slice, transform)
            output_volume = insert_slice_into_volume(output_volume, registered_slice, slice_idx, axis)
        else:
            output_volume = insert_slice_into_volume(output_volume, moving_slice, slice_idx, axis)
    
    return output_volume, translations_x, translations_y, rotations

# %%
def visualize_slice_registration_results(fixed_volume, moving_volume, 
                                       image2_aligned, translations_x, 
                                       translations_y, sample_slices=None, axis=0):
    if sample_slices is None:
        size = fixed_volume.GetLargestPossibleRegion().GetSize()
        if axis == 0:
            total_slices = size[0]
        elif axis == 1:
            total_slices = size[1]
        else: # axis == 2
            total_slices = size[2]
        sample_slices = [total_slices//4, total_slices//2, 3*total_slices//4]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    slice_indices = list(range(len(translations_x)))
    
    axes[0].plot(slice_indices, translations_x, 'b-', label='X translation')
    axes[0].set_xlabel('Slice Index')
    axes[0].set_ylabel('Translation (pixels)')
    axes[0].set_title('X Translation per Slice')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(slice_indices, translations_y, 'r-', label='Y translation')
    axes[1].set_xlabel('Slice Index')
    axes[1].set_ylabel('Translation (pixels)')
    axes[1].set_title('Y Translation per Slice')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

    fixed_array = itk.array_view_from_image(fixed_volume)
    moving_array = itk.array_view_from_image(moving_volume)
    registered_array = itk.array_view_from_image(image2_aligned)
    
    for slice_idx in sample_slices:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        if axis == 0:  # axial
            fixed_slice = fixed_array[slice_idx, :, :]
            moving_slice = moving_array[slice_idx, :, :]
            registered_slice = registered_array[slice_idx, :, :]
        elif axis == 1:  # coronal
            fixed_slice = fixed_array[:, slice_idx, :]
            moving_slice = moving_array[:, slice_idx, :]
            registered_slice = registered_array[:, slice_idx, :]
        else:  # sagittal
            fixed_slice = fixed_array[:, :, slice_idx]
            moving_slice = moving_array[:, :, slice_idx]
            registered_slice = registered_array[:, :, slice_idx]
        
        axes[0, 0].imshow(fixed_slice, cmap='gray')
        axes[0, 0].set_title(f'Fixed - Slice {slice_idx}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(moving_slice, cmap='gray')
        axes[0, 1].set_title(f'Moving - Slice {slice_idx}')
        axes[0, 1].axis('off')
        
        diff_before = fixed_slice - moving_slice
        im1 = axes[0, 2].imshow(diff_before, cmap='RdBu_r', vmin=-100, vmax=100)
        axes[0, 2].set_title('Difference Before')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(registered_slice, cmap='gray')
        axes[1, 0].set_title(f'Registered - Slice {slice_idx}')
        axes[1, 0].axis('off')
        
        checker_size = 20
        checker = np.zeros_like(fixed_slice)
        for i in range(0, checker.shape[0], checker_size*2):
            for j in range(0, checker.shape[1], checker_size*2):
                checker[i:i+checker_size, j:j+checker_size] = 1
                checker[i+checker_size:i+2*checker_size, j+checker_size:j+2*checker_size] = 1
        
        checker_img = fixed_slice * checker + registered_slice * (1 - checker)
        axes[1, 1].imshow(checker_img, cmap='gray')
        axes[1, 1].set_title('Checkerboard')
        axes[1, 1].axis('off')
        
        diff_after = fixed_slice - registered_slice
        im2 = axes[1, 2].imshow(diff_after, cmap='RdBu_r', vmin=-100, vmax=100)
        axes[1, 2].set_title('Difference After')
        axes[1, 2].axis('off')
        
        if slice_idx < len(translations_x):
            fig.suptitle(f'Slice {slice_idx} - Translation: ({translations_x[slice_idx]:.2f}, {translations_y[slice_idx]:.2f})')
        
        plt.tight_layout()
        plt.show()

# %%
def print_volume_info(volume, name):
    print(f"\n{name} volume info:")
    print(f"  Size: {volume.GetLargestPossibleRegion().GetSize()}")
    print(f"  Spacing: {volume.GetSpacing()}")
    print(f"  Origin: {volume.GetOrigin()}")
    print(f"  Direction: {volume.GetDirection()}")

# %%
def calculate_registration_metrics(fixed_image, registered_moving_image):
    diff_filter = itk.SubtractImageFilter.New(Input1=fixed_image, Input2=registered_moving_image)
    diff_filter.Update()
    diff_image = diff_filter.GetOutput()
    
    stats_filter = itk.StatisticsImageFilter.New(diff_image)
    stats_filter.Update()
    
    mse = stats_filter.GetVariance()
    mean_diff = stats_filter.GetMean()
    
    print(f"Registration quality metrics:")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Mean squared error: {mse:.6f}")
    
    return {'mse': mse, 'mean_diff': mean_diff}


# %%
fixed_image = image1
moving_image = image2

print_volume_info(fixed_image, "Fixed")
print_volume_info(moving_image, "Moving")

size = fixed_image.GetLargestPossibleRegion().GetSize()
print(f"\nVolume dimensions: {size[0]} x {size[1]} x {size[2]}")
axis = 2  # 0=axial (z), 1=coronal (y), 2=sagittal (x)

print(f"\nPerforming slice-by-slice registration along axis {axis}...")
image2_aligned, trans_x, trans_y, rotations = register_volume_slice_by_slice(
    fixed_image, 
    moving_image, 
    transform_type='translation',
    axis=axis,
    show_progress=True
)

print_volume_info(image2_aligned, "Registered")

if axis == 0:
    sample_slices = [size[0]//4, size[0]//2, 3*size[0]//4]
elif axis == 1:
    sample_slices = [size[1]//4, size[1]//2, 3*size[1]//4]
else:  # axis == 2
    sample_slices = [size[2]//4, size[2]//2, 3*size[2]//4]

print(f"Sample slices for axis {axis}: {sample_slices}")

visualize_slice_registration_results(
    fixed_image, 
    moving_image, 
    image2_aligned, 
    trans_x, 
    trans_y, 
    sample_slices,
    axis=axis
)

print("\nCalculating registration quality metrics...")
metrics = calculate_registration_metrics(fixed_image, image2_aligned)

# %% [markdown]
# # 3. Segmentation

# %%
def segment_tumor(slice, seedX, seedY, lower, upper):
    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(
        Input=slice,
        NumberOfIterations=200,
        TimeStep=0.04,
        ConductanceParameter=3
    )
    smoother.Update()
    smoothed_image = smoother.GetOutput()
    
    lower = smoothed_image.GetPixel((seedX, seedY)) - lower
    upper = smoothed_image.GetPixel((seedX, seedY)) + upper
    
    connected_threshold = itk.ConnectedThresholdImageFilter.New(smoothed_image)
    connected_threshold.SetLower(lower)
    connected_threshold.SetUpper(upper)
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetSeed((seedX, seedY))
    connected_threshold.Update()
    
    dimension = slice.GetImageDimension()

    in_type = itk.output(connected_threshold)
    output_type = itk.Image[itk.UC, dimension]
    rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.Update()
    
    return rescaler.GetOutput()


# %%
def extract_segmented_slice(volume, slice_index, axis=0, seedX=74, seedY=45, lower=150, upper=100):
    original_slice = extract_slice(volume, slice_index, axis=axis)
    segmented_slice = segment_tumor(original_slice, seedX=seedX, seedY=seedY, 
                                  lower=lower, upper=upper)
    
    return original_slice, segmented_slice

# %%
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt

seedX = 90
seedY = 70
lower_threshold = 100
upper_threshold = 50

def view_slice_range(slice_num):
    original_slice_moving, segmented_slice_moving = extract_segmented_slice(
        fixed_image, slice_index=slice_num, axis=0,
        seedX=seedX, seedY=seedY, lower=lower_threshold, upper=upper_threshold
    )
    
    original_slice_registered, segmented_slice_registered = extract_segmented_slice(
        image2_aligned, slice_index=slice_num, axis=0,
        seedX=seedX, seedY=seedY, lower=lower_threshold, upper=upper_threshold
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(itk.array_view_from_image(original_slice_moving), cmap='gray')
    axes[0, 0].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
    axes[0, 0].set_title(f'Original Fixed Slice {slice_num}')
    
    axes[0, 1].imshow(itk.array_view_from_image(segmented_slice_moving), cmap='gray')
    axes[0, 1].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
    axes[0, 1].set_title(f'Segmented Fixed Slice {slice_num}')
    
    seg_moving_array = itk.array_view_from_image(segmented_slice_moving)
    pixels_moving = np.sum(seg_moving_array > 0)
    axes[0, 2].text(0.1, 0.8, f'Fixed Image\nSlice {slice_num}', fontsize=14, fontweight='bold')
    axes[0, 2].text(0.1, 0.6, f'Segmented pixels: {pixels_moving}', fontsize=12)
    if slice_num < len(trans_x):
        axes[0, 2].text(0.1, 0.4, f'Translation X: {trans_x[slice_num]:.2f}', fontsize=10)
        axes[0, 2].text(0.1, 0.3, f'Translation Y: {trans_y[slice_num]:.2f}', fontsize=10)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(itk.array_view_from_image(original_slice_registered), cmap='gray')
    axes[1, 0].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
    axes[1, 0].set_title(f'Original Registered Slice {slice_num}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(itk.array_view_from_image(segmented_slice_registered), cmap='gray')
    axes[1, 1].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
    axes[1, 1].set_title(f'Segmented Registered Slice {slice_num}')
    axes[1, 1].axis('off')
    
    seg_registered_array = itk.array_view_from_image(segmented_slice_registered)
    pixels_registered = np.sum(seg_registered_array > 0)
    axes[1, 2].text(0.1, 0.8, f'Registered Image\nSlice {slice_num}', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.6, f'Segmented pixels: {pixels_registered}', fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'Difference: {pixels_registered - pixels_moving}', fontsize=10)
    axes[1, 2].text(0.1, 0.3, f'% Change: {((pixels_registered - pixels_moving) / max(pixels_moving, 1)) * 100:.1f}%', fontsize=10)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    plt.show()
    plt.close()
    
# interactive slider for slices 41-59
interact(view_slice_range, slice_num=IntSlider(min=41, max=59, step=1, value=50, description='Slice:'))

# %%
# # Display segmentation results for both original and registered volumes
# slices_to_show = [41, 50, 59] 
# seedX, seedY = 74, 45  # Store seed coordinates

# fig, axes = plt.subplots(3, len(slices_to_show), figsize=(20, 12))

# for i, slice_idx in enumerate(slices_to_show):
#     original_slice_moving, segmented_slice_moving = extract_segmented_slice(
#         moving_image, slice_index=slice_idx, axis=2,
#         seedX=seedX, seedY=seedY, lower=100, upper=20
#     )
    
#     original_slice_registered, segmented_slice_registered = extract_segmented_slice(
#         image2_aligned, slice_index=slice_idx, axis=2, 
#         seedX=seedX, seedY=seedY, lower=100, upper=20
#     )
    
#     # Display with seed marker
#     axes[0, i].imshow(itk.array_view_from_image(original_slice_moving), cmap='gray')
#     axes[0, i].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
#     axes[0, i].set_title(f'Original Moving Slice {slice_idx}')
#     axes[0, i].axis('off')
    
#     # Display segmented moving slice
#     axes[1, i].imshow(itk.array_view_from_image(segmented_slice_moving), cmap='gray')
#     axes[1, i].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
#     axes[1, i].set_title(f'Segmented Moving Slice {slice_idx}')
#     axes[1, i].axis('off')
    
#     # Display segmented registered slice
#     axes[2, i].imshow(itk.array_view_from_image(segmented_slice_registered), cmap='gray')
#     axes[2, i].plot(seedX, seedY, 'r+', markersize=10, markeredgewidth=2)
#     axes[2, i].set_title(f'Segmented Registered Slice {slice_idx}')
#     axes[2, i].axis('off')

# axes[0, 0].text(-0.15, 0.5, 'Original\nMoving', transform=axes[0, 0].transAxes, 
#                 rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
# axes[1, 0].text(-0.15, 0.5, 'Segmented\nMoving', transform=axes[1, 0].transAxes, 
#                 rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
# axes[2, 0].text(-0.15, 0.5, 'Segmented\nRegistered', transform=axes[2, 0].transAxes, 
#                 rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')

# plt.suptitle('Tumor Segmentation Comparison: Original vs Registered Volumes\n(Red + shows seed position)', fontsize=16)
# plt.tight_layout()
# plt.show()

# print("\nSegmentation Statistics:")
# for i, slice_idx in enumerate(slices_to_show):
#     _, seg_moving = extract_segmented_slice(moving_image, slice_idx, axis=2, seedX=seedX, seedY=seedY, lower=100, upper=20)
#     _, seg_registered = extract_segmented_slice(image2_aligned, slice_idx, axis=2, seedX=seedX, seedY=seedY, lower=100, upper=20)
    
#     seg_moving_array = itk.array_view_from_image(seg_moving)
#     seg_registered_array = itk.array_view_from_image(seg_registered)
    
#     pixels_moving = np.sum(seg_moving_array > 0)
#     pixels_registered = np.sum(seg_registered_array > 0)
    
#     print(f"Slice {slice_idx}: Moving={pixels_moving} pixels, Registered={pixels_registered} pixels")

# %% [markdown]
# # 4. Analyses

# %%
def segment_3d_volume(volume, seedX=90, seedY=70, seedZ=50, lower=100, upper=50):
    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(
        Input=volume,
        NumberOfIterations=5,
        TimeStep=0.04,
        ConductanceParameter=3
    )
    smoother.Update()
    smoothed_image = smoother.GetOutput()
    
    seed_intensity = smoothed_image.GetPixel((seedX, seedY, seedZ))
    lower_bound = seed_intensity - lower
    upper_bound = seed_intensity + upper
    
    connected_threshold = itk.ConnectedThresholdImageFilter.New(smoothed_image)
    connected_threshold.SetLower(lower_bound)
    connected_threshold.SetUpper(upper_bound)
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetSeed((seedX, seedY, seedZ))
    connected_threshold.Update()
    
    dimension = volume.GetImageDimension()
    in_type = itk.output(connected_threshold)
    output_type = itk.Image[itk.UC, dimension]
    rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.Update()
    
    return rescaler.GetOutput()

# %%
def calculate_basic_metrics(mask_image, slice_range=(41, 59)):
    spacing = mask_image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    mask_array = itk.array_view_from_image(mask_image)
    binary_mask = mask_array > 0
    
    voxel_count = np.sum(binary_mask)
    volume_mm3 = voxel_count * voxel_volume
    
    start_slice, end_slice = slice_range
    start_slice = max(0, start_slice)
    end_slice = min(binary_mask.shape[0], end_slice + 1)  # +1 because it works

    focused_mask = binary_mask[start_slice:end_slice, :, :]
    focused_voxel_count = np.sum(focused_mask)
    focused_volume_mm3 = focused_voxel_count * voxel_volume
    
    print(f"  Full volume: {voxel_count} voxels ({volume_mm3:.2f} mm³)")
    print(f"  Slices {start_slice}-{end_slice-1}: {focused_voxel_count} voxels ({focused_volume_mm3:.2f} mm³)")
    print(f"  Focused region represents {(focused_voxel_count/max(voxel_count,1)*100):.1f}% of total tumor")
    
    surface_area_mm2 = 0
    if focused_voxel_count > 0:
        # count voxels that have at least one non-tumor neighbor
        surface_voxels = 0
        for i in range(focused_mask.shape[0]):
            for j in range(focused_mask.shape[1]):
                for k in range(focused_mask.shape[2]):
                    if focused_mask[i, j, k]:
                        is_surface = False
                        neighbors = [
                            (i-1, j, k), (i+1, j, k),
                            (i, j-1, k), (i, j+1, k),
                            (i, j, k-1), (i, j, k+1)
                        ]
                        for ni, nj, nk in neighbors:
                            if (ni < 0 or ni >= focused_mask.shape[0] or
                                nj < 0 or nj >= focused_mask.shape[1] or
                                nk < 0 or nk >= focused_mask.shape[2] or
                                not focused_mask[ni, nj, nk]):
                                is_surface = True
                                break
                        if is_surface:
                            surface_voxels += 1
        
        surface_area_mm2 = surface_voxels * (spacing[0] * spacing[1])
    
    if voxel_count > 0:
        indices = np.where(binary_mask)
        full_centroid = (np.mean(indices[0]), np.mean(indices[1]), np.mean(indices[2]))
    else:
        full_centroid = (0, 0, 0)
    
    if focused_voxel_count > 0:
        focused_indices = np.where(focused_mask)
        focused_centroid = (np.mean(focused_indices[0]) + start_slice, 
                           np.mean(focused_indices[1]), 
                           np.mean(focused_indices[2]))
    else:
        focused_centroid = full_centroid
    
    return {
        'voxel_count': voxel_count,  # Full volume
        'volume_mm3': volume_mm3,    # Full volume
        'focused_voxel_count': focused_voxel_count,  # Focused region
        'focused_volume_mm3': focused_volume_mm3,    # Focused region
        'surface_area_mm2': surface_area_mm2,       # Focused region only
        'centroid': full_centroid,                 # Full volume centroid
        'focused_centroid': focused_centroid,     # Focused region centroid
        'slice_range': slice_range,
        'actual_slice_range': (start_slice, end_slice-1)
    }

# %%
def analyze_tumor_changes_minimal(fixed_image, registered_image, 
                                 seedX=90, seedY=70, seedZ=50, lower=100, upper=50,
                                 slice_range=(41, 59)):
    
    print("Performing 3D tumor segmentation...")
    
    seg_fixed = segment_3d_volume(fixed_image, seedX, seedY, seedZ, lower, upper)
    seg_registered = segment_3d_volume(registered_image, seedX, seedY, seedZ, lower, upper)
    
    print("\nCalculating metrics for FIXED volume:")
    metrics_fixed = calculate_basic_metrics(seg_fixed, slice_range)
    
    print("\nCalculating metrics for REGISTERED volume:")
    metrics_registered = calculate_basic_metrics(seg_registered, slice_range)
    
    volume_change = metrics_registered['focused_volume_mm3'] - metrics_fixed['focused_volume_mm3']
    volume_change_percent = (volume_change / metrics_fixed['focused_volume_mm3'] * 100) if metrics_fixed['focused_volume_mm3'] > 0 else 0
    
    centroid_shift = np.linalg.norm(np.array(metrics_registered['focused_centroid']) - np.array(metrics_fixed['focused_centroid']))
    
    # Dice
    mask_fixed = itk.array_view_from_image(seg_fixed) > 0
    mask_registered = itk.array_view_from_image(seg_registered) > 0
    
    start_slice, end_slice = slice_range
    start_slice = max(0, start_slice)
    end_slice = min(mask_fixed.shape[0], end_slice + 1)
    
    focused_mask_fixed = mask_fixed[start_slice:end_slice, :, :]
    focused_mask_registered = mask_registered[start_slice:end_slice, :, :]
    
    intersection = np.sum(focused_mask_fixed & focused_mask_registered)
    dice_coefficient = (2 * intersection) / (np.sum(focused_mask_fixed) + np.sum(focused_mask_registered)) if (np.sum(focused_mask_fixed) + np.sum(focused_mask_registered)) > 0 else 0
    
    # Surface area change
    surface_area_change = metrics_registered['surface_area_mm2'] - metrics_fixed['surface_area_mm2']
    surface_area_change_percent = (surface_area_change / metrics_fixed['surface_area_mm2'] * 100) if metrics_fixed['surface_area_mm2'] > 0 else 0
    
    return {
        'fixed_metrics': metrics_fixed,
        'registered_metrics': metrics_registered,
        'volume_change_mm3': volume_change,
        'volume_change_percent': volume_change_percent,
        'surface_area_change_mm2': surface_area_change,
        'surface_area_change_percent': surface_area_change_percent,
        'centroid_shift_voxels': centroid_shift,
        'dice_coefficient': dice_coefficient,
        'slice_range': slice_range,
        'segmentations': {
            'fixed': seg_fixed,
            'registered': seg_registered
        }
    }

# %%
def analyze_intensity_changes_minimal(fixed_image, registered_image, seg_fixed, seg_registered):
    
    fixed_array = itk.array_view_from_image(fixed_image)
    registered_array = itk.array_view_from_image(registered_image)
    mask_fixed = itk.array_view_from_image(seg_fixed) > 0
    mask_registered = itk.array_view_from_image(seg_registered) > 0
    
    intensities_fixed = fixed_array[mask_fixed]
    intensities_registered = registered_array[mask_registered]
    
    stats_fixed = {
        'mean': np.mean(intensities_fixed),
        'std': np.std(intensities_fixed)
    }
    
    stats_registered = {
        'mean': np.mean(intensities_registered),
        'std': np.std(intensities_registered)
    }
    
    return {
        'fixed_stats': stats_fixed,
        'registered_stats': stats_registered,
        'mean_change': stats_registered['mean'] - stats_fixed['mean'],
        'intensities_fixed': intensities_fixed,
        'intensities_registered': intensities_registered
    }

# %%
def analyze_slice_by_slice_numpy(seg_fixed, seg_registered, axis=0):
    mask_fixed = itk.array_view_from_image(seg_fixed) > 0
    mask_registered = itk.array_view_from_image(seg_registered) > 0
    
    if axis == 0:
        num_slices = mask_fixed.shape[0]
        slice_func = lambda i: (mask_fixed[i, :, :], mask_registered[i, :, :])
    elif axis == 1:
        num_slices = mask_fixed.shape[1]
        slice_func = lambda i: (mask_fixed[:, i, :], mask_registered[:, i, :])
    else:
        num_slices = mask_fixed.shape[2]
        slice_func = lambda i: (mask_fixed[:, :, i], mask_registered[:, :, i])
    
    slice_metrics = []
    
    for i in range(num_slices):
        slice_fixed, slice_registered = slice_func(i)
        
        area_fixed = np.sum(slice_fixed)
        area_registered = np.sum(slice_registered)
        area_change = area_registered - area_fixed
        
        # Dice coefficient for this slice
        intersection = np.sum(slice_fixed & slice_registered)
        dice = (2 * intersection) / (area_fixed + area_registered) if (area_fixed + area_registered) > 0 else 0
        
        slice_metrics.append({
            'slice': i,
            'area_fixed': area_fixed,
            'area_registered': area_registered,
            'area_change': area_change,
            'dice': dice
        })
    
    return slice_metrics

# %%
def analyze_slice_by_slice_minimal(seg_fixed, seg_registered):
    mask_fixed = itk.array_view_from_image(seg_fixed) > 0
    mask_registered = itk.array_view_from_image(seg_registered) > 0
    
    num_slices = mask_fixed.shape[0]
    
    slice_stats = []
    for i in range(num_slices):
        slice_fixed = mask_fixed[i, :, :]
        slice_registered = mask_registered[i, :, :]
        
        area_fixed = np.sum(slice_fixed)
        area_registered = np.sum(slice_registered)
        area_change = area_registered - area_fixed
        
        # Only calculate dice if there's something in at least one slice
        if area_fixed > 0 or area_registered > 0:
            intersection = np.sum(slice_fixed & slice_registered)
            dice = (2 * intersection) / (area_fixed + area_registered) if (area_fixed + area_registered) > 0 else 0
        else:
            dice = 1.0  # Perfect match for empty slices
        
        slice_stats.append({
            'slice': i,
            'area_fixed': area_fixed,
            'area_registered': area_registered,
            'area_change': area_change,
            'dice': dice
        })
    
    return slice_stats

# %%
def debug_change_map(seg_fixed, seg_registered):
    mask_fixed = itk.array_view_from_image(seg_fixed) > 0
    mask_registered = itk.array_view_from_image(seg_registered) > 0
    
    print(f"Fixed mask: {np.sum(mask_fixed)} voxels")
    print(f"Registered mask: {np.sum(mask_registered)} voxels")
    print(f"Intersection: {np.sum(mask_fixed & mask_registered)} voxels")
    print(f"Union: {np.sum(mask_fixed | mask_registered)} voxels")
    
    change_map = np.zeros_like(mask_fixed, dtype=np.uint8)
    change_map[mask_fixed & ~mask_registered] = 1
    change_map[mask_fixed & mask_registered] = 2
    change_map[~mask_fixed & mask_registered] = 3
    
    unique_values, counts = np.unique(change_map, return_counts=True)
    print(f"Change map values: {dict(zip(unique_values, counts))}")
    print(f"0=background, 1=shrinkage, 2=stable, 3=growth")
    
    return change_map

# %%
def create_change_map_minimal(seg_fixed, seg_registered):
    mask_fixed = itk.array_view_from_image(seg_fixed) > 0
    mask_registered = itk.array_view_from_image(seg_registered) > 0
    
    change_map = np.zeros_like(mask_fixed, dtype=np.uint8)
    change_map[mask_fixed & ~mask_registered] = 1
    change_map[mask_fixed & mask_registered] = 2
    change_map[~mask_fixed & mask_registered] = 3
    
    return change_map

# %%
def visualize_results_minimal(analysis_results, intensity_results):
    print("\n=== NUMERICAL ANALYSIS RESULTS ===")
    slice_range = analysis_results.get('slice_range', (41, 59))
    
    # Volume analysis - show both full and focused
    volume_fixed_full = analysis_results['fixed_metrics']['volume_mm3']
    volume_registered_full = analysis_results['registered_metrics']['volume_mm3']
    volume_fixed_focused = analysis_results['fixed_metrics']['focused_volume_mm3']
    volume_registered_focused = analysis_results['registered_metrics']['focused_volume_mm3']
    
    volume_change = analysis_results['volume_change_mm3']
    volume_change_percent = analysis_results['volume_change_percent']
    
    print(f"\n📊 VOLUME ANALYSIS (Focused on slices {slice_range[0]}-{slice_range[1]}):")
    print(f"  Full Volume - Time 1: {volume_fixed_full:.2f} mm³")
    print(f"  Full Volume - Time 2: {volume_registered_full:.2f} mm³")
    print(f"  Focused Volume - Time 1: {volume_fixed_focused:.2f} mm³")
    print(f"  Focused Volume - Time 2: {volume_registered_focused:.2f} mm³")
    print(f"  Focused Volume Change: {volume_change:.2f} mm³")
    print(f"  Focused Percentage Change: {volume_change_percent:.2f}%")
    
    # Surface area analysis
    surface_fixed = analysis_results['fixed_metrics']['surface_area_mm2']
    surface_registered = analysis_results['registered_metrics']['surface_area_mm2']
    surface_change = analysis_results.get('surface_area_change_mm2', 0)
    surface_change_percent = analysis_results.get('surface_area_change_percent', 0)
    
    print(f"\n🔲 SURFACE ANALYSIS (Slices {slice_range[0]}-{slice_range[1]} only):")
    print(f"  Surface Area - Time 1: {surface_fixed:.2f} mm²")
    print(f"  Surface Area - Time 2: {surface_registered:.2f} mm²")
    print(f"  Surface Area Change: {surface_change:.2f} mm² ({surface_change_percent:.2f}%)")
    
    dice = analysis_results['dice_coefficient']
    print(f"\n🎯 OVERLAP ANALYSIS (Focused region):")
    print(f"  Dice Coefficient: {dice:.4f}")
    print(f"  Overlap Percentage: {dice*100:.2f}%")
    
    centroid_shift = analysis_results['centroid_shift_voxels']
    print(f"\n📍 SPATIAL ANALYSIS (Focused centroids):")
    print(f"  Centroid Shift: {centroid_shift:.2f} voxels")
    print(f"  Time 1 Focused Centroid: {analysis_results['fixed_metrics']['focused_centroid']}")
    print(f"  Time 2 Focused Centroid: {analysis_results['registered_metrics']['focused_centroid']}")
    
    mean_change = intensity_results['mean_change']
    print(f"\n💡 INTENSITY ANALYSIS:")
    print(f"  Mean Intensity Change: {mean_change:.2f}")
    print(f"  Time 1 Mean: {intensity_results['fixed_stats']['mean']:.2f}")
    print(f"  Time 2 Mean: {intensity_results['registered_stats']['mean']:.2f}")
    print(f"  Time 1 Std Dev: {intensity_results['fixed_stats']['std']:.2f}")
    print(f"  Time 2 Std Dev: {intensity_results['registered_stats']['std']:.2f}")
    
    slice_stats = analyze_slice_by_slice_minimal(
        analysis_results['segmentations']['fixed'],
        analysis_results['segmentations']['registered']
    )
    
    focused_slice_stats = [s for s in slice_stats if slice_range[0] <= s['slice'] <= slice_range[1]]
    
    focused_area_changes = [s['area_change'] for s in focused_slice_stats]
    focused_dice_scores = [s['dice'] for s in focused_slice_stats if s['area_fixed'] > 0 or s['area_registered'] > 0]
    
    focused_slices_with_tumor = [s for s in focused_slice_stats if s['area_fixed'] > 0 or s['area_registered'] > 0]
    
    print(f"\n🔍 SLICE-BY-SLICE ANALYSIS (Slices {slice_range[0]}-{slice_range[1]}):")
    print(f"  Focused Slices Analyzed: {len(focused_slice_stats)}")
    print(f"  Focused Slices with Tumor: {len(focused_slices_with_tumor)}")
    print(f"  Mean Area Change per Focused Slice: {np.mean(focused_area_changes):.2f} voxels")
    print(f"  Std Dev Area Change (Focused): {np.std(focused_area_changes):.2f} voxels")
    print(f"  Max Growth in Focused Slice: {np.max(focused_area_changes):.0f} voxels")
    print(f"  Max Shrinkage in Focused Slice: {np.min(focused_area_changes):.0f} voxels")
    if focused_dice_scores:
        print(f"  Mean Dice Score (Focused tumor slices): {np.mean(focused_dice_scores):.4f}")
    
    focused_growth_slices = len([s for s in focused_slice_stats if s['area_change'] > 0])
    focused_shrinkage_slices = len([s for s in focused_slice_stats if s['area_change'] < 0])
    focused_stable_slices = len([s for s in focused_slice_stats if s['area_change'] == 0])
    
    print(f"\n📈 FOCUSED SLICE CHANGE BREAKDOWN:")
    print(f"  Slices with Growth: {focused_growth_slices}")
    print(f"  Slices with Shrinkage: {focused_shrinkage_slices}")
    print(f"  Stable Slices: {focused_stable_slices}")
    
    focused_slices_with_changes = [s for s in focused_slice_stats if s['area_change'] != 0]
    if focused_slices_with_changes:
        print(f"\n📋 FOCUSED SLICES WITH SIGNIFICANT CHANGES:")
        for s in sorted(focused_slices_with_changes, key=lambda x: abs(x['area_change']), reverse=True)[:5]:
            change_type = "Growth" if s['area_change'] > 0 else "Shrinkage"
            print(f"  Slice {s['slice']}: {change_type} of {abs(s['area_change'])} voxels (Dice: {s['dice']:.3f})")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Show focused volume comparison
    axes[0].bar(['Time 1', 'Time 2'], [volume_fixed_focused, volume_registered_focused], 
                color=['blue', 'red'], alpha=0.7)
    axes[0].set_ylabel('Volume (mm³)')
    axes[0].set_title(f'Focused Volume (Slices {slice_range[0]}-{slice_range[1]})\n{volume_fixed_focused:.1f} → {volume_registered_focused:.1f} mm³ ({volume_change_percent:+.1f}%)')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    axes[0].text(0, volume_fixed_focused/2, f'{volume_fixed_focused:.1f}', ha='center', va='center', fontweight='bold')
    axes[0].text(1, volume_registered_focused/2, f'{volume_registered_focused:.1f}', ha='center', va='center', fontweight='bold')
    
    # 2. Show area changes only for focused slices
    focused_slice_indices = [s['slice'] for s in focused_slice_stats]
    focused_area_changes = [s['area_change'] for s in focused_slice_stats]
    
    colors = ['red' if change > 0 else 'blue' if change < 0 else 'gray' for change in focused_area_changes]
    axes[1].bar(focused_slice_indices, focused_area_changes, color=colors, alpha=0.7)
    axes[1].set_xlabel('Slice Index')
    axes[1].set_ylabel('Area Change (voxels)')
    axes[1].set_title(f'Area Change per Slice (Focused Range: {slice_range[0]}-{slice_range[1]})')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    if volume_change_percent > 5:
        print(f"\n⚠️  CLASSIFICATION: SIGNIFICANT TUMOR GROWTH in focused region ({volume_change_percent:.1f}%)")
    elif volume_change_percent < -5:
        print(f"\n✅ CLASSIFICATION: SIGNIFICANT TUMOR SHRINKAGE in focused region ({volume_change_percent:.1f}%)")
    else:
        print(f"\n📊 CLASSIFICATION: STABLE TUMOR SIZE in focused region ({volume_change_percent:.1f}%)")
    
    return fig

# %%
def run_minimal_tumor_analysis(fixed_image, registered_image, 
                              seedX=90, seedY=70, seedZ=50, 
                              lower=100, upper=50,
                              slice_range=(41, 59)):
    print("=== MINIMAL TUMOR ANALYSIS ===")
    print(f"Focusing surface analysis on slices {slice_range[0]} to {slice_range[1]}")
    
    analysis_results = analyze_tumor_changes_minimal(fixed_image, registered_image, 
                                                    seedX, seedY, seedZ, lower, upper,
                                                    slice_range)
    
    intensity_results = analyze_intensity_changes_minimal(
        fixed_image, registered_image,
        analysis_results['segmentations']['fixed'],
        analysis_results['segmentations']['registered']
    )
    
    fig = visualize_results_minimal(analysis_results, intensity_results)
    
    print(f"\n=== FOCUSED SUMMARY (Slices {slice_range[0]}-{slice_range[1]}) ===")
    print(f"Focused Volume Change: {analysis_results['volume_change_mm3']:.1f} mm³ ({analysis_results['volume_change_percent']:.1f}%)")
    print(f"Surface Area Change: {analysis_results.get('surface_area_change_mm2', 0):.1f} mm² ({analysis_results.get('surface_area_change_percent', 0):.1f}%)")
    print(f"Dice Coefficient: {analysis_results['dice_coefficient']:.3f}")
    print(f"Centroid Shift: {analysis_results['centroid_shift_voxels']:.1f} voxels")
    print(f"Mean Intensity Change: {intensity_results['mean_change']:.1f}")
    
    return analysis_results, intensity_results

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itk

SEED_X = 90
SEED_Y = 70  
SEED_Z = 50
LOWER_THRESHOLD = 100
UPPER_THRESHOLD = 50
SLICE_RANGE = (41, 59)

analysis_results, intensity_results = run_minimal_tumor_analysis(
    fixed_image, 
    image2_aligned, 
    seedX=SEED_X, 
    seedY=SEED_Y, 
    seedZ=SEED_Z,
    lower=LOWER_THRESHOLD, 
    upper=UPPER_THRESHOLD,
    slice_range=SLICE_RANGE
)

# %%
def create_brain_contour(volume, threshold_low=200, threshold_high=1000, 
                        closing_radius=3, opening_radius=2):
    threshold_filter = itk.BinaryThresholdImageFilter.New(
        Input=volume,
        LowerThreshold=threshold_low,
        UpperThreshold=threshold_high,
        InsideValue=255,
        OutsideValue=0
    )
    threshold_filter.Update()
    thresholded = threshold_filter.GetOutput()

    structuring_element = itk.FlatStructuringElement[3].Ball(closing_radius)
    closing_filter = itk.BinaryMorphologicalClosingImageFilter.New(
        Input=thresholded,
        Kernel=structuring_element,
        ForegroundValue=255
    )
    closing_filter.Update()
    closed = closing_filter.GetOutput()
    
    structuring_element_open = itk.FlatStructuringElement[3].Ball(opening_radius)
    opening_filter = itk.BinaryMorphologicalOpeningImageFilter.New(
        Input=closed,
        Kernel=structuring_element_open,
        ForegroundValue=255
    )
    opening_filter.Update()
    brain_mask = opening_filter.GetOutput()
    
    return brain_mask


# %%
brain_contour = create_brain_contour(fixed_image, threshold_low=200, threshold_high=1000)

# %%
import math

class ColorAnimator:
    def __init__(self, actor, render_window, get_coordinate, text_actor=None, prefix=''):
        self.actor = actor
        self.render_window = render_window
        self.get_coordinate = get_coordinate
        self.text_actor = text_actor
        self.prefix = prefix
        
    def animate_callback(self, caller, event):
        if self.text_actor:
            new_text = f"{self.prefix} | Opacity: {self.get_coordinate():.2f}"
            self.text_actor.SetInput(new_text)
        
        new_opacity = self.get_coordinate()
        
        actor_property = self.actor.GetProperty()
        actor_property.SetOpacity(new_opacity)

        self.render_window.Render()

# %%
from vtk.util import numpy_support
import math

def create_simple_3d_viewer(seg_fixed, seg_registered, show_window=True):
    def itk_to_slice(itkImage):
        source = itk.vtk_image_from_image(itkImage)
        
        contour = vtk.vtkContourFilter()
        contour.SetInputData(source)
        contour.SetValue(25, 1)

        contourMapper = vtk.vtkPolyDataMapper()

        contourMapper.SetInputConnection(contour.GetOutputPort())
        contourMapper.ScalarVisibilityOff()

        contourActor = vtk.vtkActor()

        contourActor.SetMapper(contourMapper)
        
        return contourActor
    
    seg_fixed_actor = itk_to_slice(seg_fixed)
    seg_fixed_actor.GetProperty().SetColor(0.0, 0.0, 1.0)
    seg_fixed_actor.GetProperty().SetOpacity(0.5) 
    
    seg_registered_actor = itk_to_slice(seg_registered)
    seg_registered_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    
    brain_contour_actor = itk_to_slice(brain_contour)
    brain_contour_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
    brain_contour_actor.GetProperty().SetOpacity(0.1)
    
    renderer = vtk.vtkRenderer()
    renderer.AddActor(seg_fixed_actor)
    renderer.AddActor(seg_registered_actor)
    renderer.AddActor(brain_contour_actor)
    renderer.SetBackground(0.1, 0.1, 0.1)
        
    textRedTumorActor = vtk.vtkTextActor()
    textRedTumorActor.SetTextScaleModeToNone()
    textRedTumorActor.SetPosition(10, 40)
    textRedTumorActor.SetInput("Tumor at first visit")
    textRedTumorActor.GetTextProperty().SetFontSize(24)
    textRedTumorActor.GetTextProperty().SetColor(0.0, 0.0, 1.0)
    
    textBlueTumorActor = vtk.vtkTextActor()
    textBlueTumorActor.SetTextScaleModeToNone()
    textBlueTumorActor.SetPosition(10, 10)
    textBlueTumorActor.SetInput("Tumor at second visit")
    textBlueTumorActor.GetTextProperty().SetFontSize(24)
    textBlueTumorActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
    
    textInstructions = vtk.vtkTextActor()
    textInstructions.SetTextScaleModeToNone()
    textInstructions.SetPosition(10, 70)
    textInstructions.SetInput("Move vertically and horizontally to change opacity of both tumors")
    textInstructions.GetTextProperty().SetFontSize(18)
    textInstructions.GetTextProperty().SetColor(1.0, 1.0, 1.0)

    renderer.AddActor(textBlueTumorActor)
    renderer.AddActor(textRedTumorActor)
    renderer.AddActor(textInstructions)
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1080, 720)
    render_window.SetFullScreen(True)
    render_window.SetWindowName("3D Tumor Visualization")
    
    renderer.ResetCamera()
    
    if show_window:
        render_window.Render()
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.Initialize()
            
        def get_coordinate_x():
            x, _ = interactor.GetEventPosition()
            normalized_x = x / render_window.GetSize()[0]
            
            if normalized_x < 0.2:
                return 0.0
            elif normalized_x > 0.8:
                return 0.9999
            else:
                log_min = math.log10(0.01) 
                log_max = math.log10(1.0)
                log_value = log_min + normalized_x * (log_max - log_min)
                return (10 ** log_value - 0.01) / (1.0 - 0.01)
                    
        def get_coordinate_y():
            _,y = interactor.GetEventPosition()
            normalized_y = y / render_window.GetSize()[1]
            if normalized_y < 0.2:
                return 0.0
            elif normalized_y > 0.8:
                return 0.9999
            else:
                linear_value = (normalized_y - 0.2) / 0.6
                # Convert to logarithmic scale (0 to 1 range)
                if linear_value <= 0:
                    return 0.0
                else:
                    # Map linear [0,1] to log scale [0,1]
                    # Using log10 with a range from 0.01 to 1 to avoid log(0)
                    log_min = math.log10(0.01) 
                    log_max = math.log10(1.0)
                    log_value = log_min + linear_value * (log_max - log_min)
                    return (10 ** log_value - 0.01) / (1.0 - 0.01)
   
        animator1 = ColorAnimator(seg_fixed_actor, render_window, get_coordinate_x, textRedTumorActor, prefix='Tumor at first visit')
        animator2 = ColorAnimator(seg_registered_actor, render_window, get_coordinate_y, textBlueTumorActor, prefix='Tumor at second visit')
        
        interactor.AddObserver(vtk.vtkCommand.MouseMoveEvent, animator1.animate_callback)
        interactor.AddObserver(vtk.vtkCommand.MouseMoveEvent, animator2.animate_callback)
        
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.SetRenderWindow(render_window)
        
        interactor.Start()
    
    return render_window

def show_tumor_3d(analysis_results, show_interactive=True):
    
    seg_fixed = analysis_results['segmentations']['fixed']
    seg_registered = analysis_results['segmentations']['registered']
    
    render_window = create_simple_3d_viewer(seg_fixed, seg_registered, show_interactive)
    
    return render_window

print("\n=== 3D TUMOR VISUALIZATION ===")
print("Loading...")
show_tumor_3d(analysis_results)