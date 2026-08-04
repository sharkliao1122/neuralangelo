# COLMAP mesh organization and proxy record

Date: 2026-07-30

## Folder organization

- `two_videos/`: files whose dataset name is `twovideos`
- `non_smooth/`: files whose dataset name is `nonsmooth_ds10_1600`
- `kengdong/`: files whose dataset name is `kengdong_960`

Only top-level `.ply` files in `output_colmap` were moved. Existing dataset
subfolders were not changed.

## Interactive proxy parameters

- Method: Open3D quadric mesh decimation
- Target triangle count: `1,500,000`
- Output format: binary PLY
- Preserved attributes: vertex RGB colors
- Recomputed attributes: vertex normals
- Not preserved: the COLMAP Poisson `value` scalar
- Original meshes: preserved without modification

## Results

### two_videos

- Original: `two_videos/meshed-poisson_twovideos.ply`
  - Vertices: `18,893,432`
  - Faces: `37,728,379`
  - Size: `962,629,534` bytes
- Proxy: `two_videos/meshed-poisson_twovideos_proxy_1500k.ply`
  - Vertices: `757,048`
  - Faces: `1,500,000`
  - Size: `58,109,775` bytes
  - Vertex colors: verified
  - Vertex normals: verified
  - Edge manifold: verified
  - Watertight: no

### non_smooth

- Original: `non_smooth/meshed-poisson-nonsmooth_ds10_1600.ply`
  - Vertices: `12,250,664`
  - Faces: `24,468,527`
  - Size: `624,259,310` bytes
- Proxy: `non_smooth/meshed-poisson-nonsmooth_ds10_1600_proxy_1500k.ply`
  - Vertices: `754,542`
  - Faces: `1,499,999`
  - Size: `57,981,956` bytes
  - Vertex colors: verified
  - Vertex normals: verified
  - Edge manifold: verified
  - Watertight: no

## Reusable command pattern

Run in Anaconda Prompt with the `neuralangelo_5080` environment:

```bat
conda activate neuralangelo_5080
python C:\research\neuralangelo_test\projects\neuralangelo\scripts\decimate_colored_mesh_open3d.py --input INPUT_MESH.ply --output OUTPUT_PROXY.ply --target_faces 1500000
```
