# DATA_STRATEGY.md

## Core idea

We do not have large expert robot demos and we do not have hardware yet.

Therefore, HomeBrain uses a teacher-student strategy:
1. Collect or load real indoor visual sequences.
2. Run large foundation teachers offline.
3. Save pseudo-labels and features.
4. Train a smaller SpatialMemoryNet student.
5. Evaluate using replay and measurable metrics.
6. Later attach real robot logs to the same pipeline.

## Data sources by purpose

### Real indoor visual distribution

Use public indoor datasets and self-collected phone/robot-height video to expose the model to real homes, clutter, lighting, furniture, cords, humans, pets, and motion blur.

Purpose:
- perception
- geometry priors
- dynamic object handling
- uncertainty detection

### Public RGB-D / geometry datasets

Purpose:
- image-to-depth / point-map supervision
- local BEV labels
- pose delta supervision
- place recognition/keyframes later

License must be audited before commercial use.

### Dynamic indoor datasets

Purpose:
- moving humans/pets/object-like dynamics
- do-not-permanently-map temporary obstacles
- dynamic risk labels

### Simulation

Use sim only where real videos cannot provide action-conditioned labels:
- candidate action outcomes
- coverage tasks
- collision/recovery pretraining
- synthetic rare cases

Do not try to build a perfect simulator before training useful components.

### Future robot logs

Once hardware exists, every run becomes data:
- normal coverage
- stuck events
- near-collisions
- low light
- moved furniture
- failed docking
- human intervention

## First data artifact format

For each sequence:

```text
sequence_id/
  manifest.json
  frames/
    000000.jpg
    ...
  events.jsonl.zst or events.jsonl.gz
  teachers/
    teacher_name/
      metadata.json
      depth.npy
      confidence.npy
      features.npy
      masks.npy
  eval/
    metrics.json
    overlay.mp4
```

Codex may choose a simpler compressed JSON/NPZ format initially, but the format must be deterministic and documented.
