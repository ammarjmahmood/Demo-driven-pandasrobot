# Phone data collection: two-hand tabletop demonstrations

## Setup

- Mount the phone above a plain, well-lit table. Keep it stationary and record landscape video at 1080p/30 fps or better.
- Put four thick black X markers at the corners of a roughly 50 × 35 cm working area, like the generated clip. Measure width and height **between the marker centers**, in centimetres. Keep the markers visible near the four image corners.
- Use a saturated red block and a saturated blue block, ideally about 4 cm across. Colored tape on small boxes is fine. Draw matching colored **rings** about 8 cm across on white paper. The current extractor distinguishes solid blocks from hollow target rings.
- Start each block clearly outside its ring. For the first clip keep red on the left and blue on the right, with at least 15 cm between their paths. Put both targets slightly nearer the camera's bottom edge than the starting blocks.
- Keep both hands, both objects, and both targets visible throughout each clip. Avoid face, personal documents, and reflective objects in frame.

## Record

Start with **one clip**, about 8–12 seconds:

1. Keep both hands completely clear of the blocks and rings for the first two seconds.
2. Pick up the two blocks with separate hands, lift slightly, and place each in its matching ring.
3. Remove your hands and hold the final view for two seconds.

Two more clips are useful after the first works: change the initial positions in one, and change the targets in the other. Keep one clip untouched until we evaluate it. There is no need to collect 20–30 clips before validating this pipeline. More footage would be needed for a credible learned-model experiment later.

Keep the phone still throughout each recording. Please report real mistakes rather than editing them out. A later crossing-path clip can test resource selection, but the first clip should have separate paths.

## Metadata to note

For each clip: filename, date, workspace dimensions between marker centers, approximate block dimensions, and whether each object reached the correct target. The current simulation uses 4 cm cubes; differing real shapes will need an explicit model adjustment.

## Share with me

Place clips under `data/raw/` in this workspace, or provide the directory path. Do not upload a large batch to GitHub yet. One sample clip is enough to begin calibrating tracking.
