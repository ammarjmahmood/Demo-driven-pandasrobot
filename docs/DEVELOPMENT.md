# Development log and evidence

## Stage 1 — Gemini video as an engineering fixture

**Status: implemented and tested on 6 October 2026.** The applicant supplied a ten-second Gemini-generated video showing two hands moving a red block and a blue block to matching colored rings. The video has 240 frames at 24 fps and a resolution of 1280 × 720.

We used this clip to develop the full pipeline before collecting real-world demonstrations. The generated scene provided a controllable task layout and did not require a robot or a large training run. It is explicitly labeled as generated wherever provenance is recorded.

### Problems found and changes made

1. **The first color tracker selected target rings rather than blocks.** It also detected skin as red. The replacement extractor uses a stronger saturation threshold, restricts detection to the marked tabletop, separates rings by their contour holes, removes their static strokes during tracking, and uses size and position continuity.
2. **Hands obscured the middle of the moves.** We kept tracking gaps and heuristic confidence visible. The robot task uses clean start positions and visible target rings; the pipeline does not invent a recovered hand trajectory.
3. **The first two-arm rollout placed both blocks but its hands touched.** We rotated the grasp orientation so the finger-opening direction follows the table's forward axis. The revised rollout completed without detected inter-arm contact. Overall success now also requires passing the implemented robot-contact checks.
4. **The initial planner did not actually honor dependency order.** Scheduling now uses a topological ordering and rejects cycles. Tests cover reversed dependencies, crossing paths, recruitment penalties, and invalid coordinates.

### Recorded result

The saved evidence is in [generated_baseline.json](results/generated_baseline.json). It includes the input hash, extracted task, results for both arm counts, and the layout perturbation checks.

| Configuration | Simulated duration | Both objects placed | Inter-arm contact | Nonbase robot/table contact |
|---|---:|---|---|---|
| One Panda | 19.6 s | Yes | Not applicable | None recorded |
| Two Pandas | 10.2 s | Yes | None recorded | None recorded |

The red track exceeded the area-consistency threshold of 0.5 on 87.5% of frames, and the blue track on 71.7%. These are heuristic tracking-coverage figures, not detection accuracy against annotated ground truth.

Three further scene variants independently perturbed every start and goal coordinate by up to 1 cm. Both arm configurations passed each variant: six additional successful rollouts. This is a small deterministic robustness check around one layout, not a generalization benchmark or evidence about real-video perception.

The controller's fixed phase durations explain the parallel speedup. A larger set of tasks that sometimes benefits from a second arm and sometimes does not is needed to evaluate resource selection meaningfully.

## Stage 2 — Applicant-recorded real-world footage

**Status: planned; no results yet.** Record a similar tabletop task with real objects and measured workspace dimensions. Begin with one clip to inspect calibration and tracking before collecting a larger dataset.

The first real clip will be a development clip. If we tune thresholds or geometry to it, we will say so. Reserve additional recordings and layouts for evaluation without adjusting the system after seeing their outcomes. Keep the source provenance and results distinct from the generated-video stage.

Other objects are welcome, but any changed shape, scale, appearance, or grasp strategy must be represented in perception and simulation. See [the object-transition notes](DATA_PROVENANCE.md#using-different-real-objects).

## Stage 3 — Stronger comparison and possible learned component

**Status: proposed, not implemented.** Useful next experiments include a crossing-path task that favors serialization, resource-cost sensitivity, and failure recovery. A compact learned predictor can be added if it answers a specific question and beats an appropriate baseline. The current project has not trained a VLA/VAM or implemented language-agent negotiation.

## Reproduction scope

The repository includes the labeled Gemini source clip, a portable extracted task, and a comparison video. The full generated-data pipeline can be reproduced with the bundled clip; the controller and physics can also be rerun directly from the extracted task. The asset downloader pins the third-party robot model. Raw personal videos and bulk output directories remain ignored by Git; selected real-data submission examples will be included when ready.
