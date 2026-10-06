# How the baseline works

## What it does

The prototype turns a particular kind of tabletop video into two object-transfer tasks, then executes those tasks with either one or two Panda manipulators in MuJoCo. The input video changes where the simulated objects start and where the robot places them. The current pipeline transfers endpoints rather than reproducing a human hand trajectory.

It uses geometric computer vision, explicit scheduling rules, inverse kinematics, and physics simulation. No VLA, VAM, VLM, or learned world model has been trained or called. The word "demonstration" describes the input; it does not imply imitation-learning training.

## 1. Separate objects from targets

`src/extract.py` reads a clean opening frame and finds saturated red and blue image regions in HSV color space. A colored ring has a large hole inside its contour; a solid block does not. This distinguishes each block from its target. The small colored disk at a target's center is excluded from the opening-frame block search because it lies inside the ring.

Four dark corner markers identify the working rectangle. A user-supplied calibration can replace automatic marker detection. This is designed for a fixed overhead view with visible markers and known object colors, rather than arbitrary video.

The per-frame tracker removes the static ring strokes, compares candidate blob sizes with the initial block, and limits jumps from its previous location. It records missing detections and an area-consistency score. The score is a heuristic, not a calibrated probability or an accuracy measurement. A hand covering a block can still bias its detected center.

## 2. Map pixels into a tabletop

A homography maps the four marker centers to a rectangle with supplied width and height in metres. The generated-video test used an assumed 0.50 × 0.35 m rectangle. The real-data stage must use measured distances between marker centers.

This is a planar approximation. It does not recover camera intrinsics, object depth, fingertip pose, or a three-dimensional human trajectory. The visible center of a raised block is not exactly its contact point on the table, which is a source of calibration error.

The opening frame supplies block positions and target-ring centers. A color-change heuristic at the target in the final frame checks whether the demonstrated placement appears supported. That check is reported in `task.json`; it does not certify grasp or task success. The intended target remains the ring center even if that heuristic is inconclusive.

## 3. Represent the task

The task JSON contains:

- Each object's identity, start, and goal, in both pixels and tabletop metres.
- Supplied workspace dimensions and detected marker positions.
- A source provenance label and SHA-256 hash of the original video.
- Tracking-quality diagnostics and explicit assumptions.
- Optional dependencies such as `["blue", "red"]`, meaning blue must move first.

The baseline assumes the two separate block transfers are independent. It does not infer that assumption from language, hand identities, observed simultaneity, or physical causality. The human video supplies the geometric instance of a known task type.

## 4. Choose the number of arms

`src/plan.py` constructs schedules for one and two arms. One arm handles both transfers sequentially. Two arms can handle one transfer each in parallel if there are no supplied dependencies and the sampled planar transport corridors are at least 12 cm apart.

The planner compares the schedule duration plus an explicit second-arm recruitment penalty. The current default penalty is 1.5 seconds and is a design parameter, not a measured networking or setup delay. The simulator does not physically add a robot partway through the run: it constructs the selected one- or two-arm scene at initialization. The end-to-end command runs both configurations for comparison, even though the planner also records its preferred choice.

Durations come from the scripted controller's phase timings. Every object move currently has the same phase budget. This makes the timing estimates inspectable but is not a learned efficiency model. The two-arm speedup on an independent two-object task is expected from parallel execution.

Dependencies are topologically ordered, and cycles are rejected. Overlapping transport corridors cause serialization. The corridor test concerns tabletop paths; it is not a full collision-aware planner for robot links, approach motion, or all possible objects.

The readable coordination message is generated from these actual assignments and scheduling constraints. There are no independently reasoning language agents or open-ended negotiations.

## 5. Execute a physical robot grasp

`src/scene.py` composes one or two Panda models from the pinned MuJoCo Menagerie assets, a tabletop plane, two 4 cm free-body cubes, and non-colliding visual target disks. The one-arm base is centered; two-arm bases sit on either side of the centerline. A human recording is mapped into this chosen robot workspace rather than copied as an exact digital twin.

`src/simulate.py` commands this sequence for each transfer:

1. Move above the object with the gripper open.
2. Lower the fingers around it and close the gripper.
3. Lift, translate above the target, and lower.
4. Open the gripper, retreat, and let the block settle.

An inverse-kinematics solver uses MuJoCo Jacobians to find joint configurations for Cartesian waypoints. Joint actuator targets are smoothly interpolated between waypoints, and the simulation advances with a 2 ms physics step. Arm bias forces are compensated. Finger contact and friction hold the objects; no runtime object teleportation or grasp attachment is used.

The controller reads exact simulated block positions at the start of each transfer. Consequently, this is not an image-only closed-loop robot policy. The video establishes the task geometry, and the simulator supplies execution state.

## 6. Check what happened

A block passes the current placement test when:

- Its final horizontal distance from its target is below 3 cm.
- Its final center height is between 1 cm and 4.5 cm.
- Its center rose above 8 cm during execution, providing evidence of a lift.

Overall success additionally requires no logged contact between the two arms or between a nonbase robot part and the tabletop. This is the set of collision checks implemented, not a proof of universal collision avoidance. The final pose is checked after a settling interval; velocity is not a separate success criterion.

The outputs include physical state results, phase events, contact-pair logs, annotated rollouts, and a comparison video. The three-panel video uses a common elapsed clock, with each panel holding its final frame when finished. It does not align human and robot action phases or establish human-speed equivalence.

## What would make this a learned-model experiment

A future model needs a defined target and comparison—for example, predicting action completion from images or predicting whether proposed concurrent motions will interfere. It must be evaluated against this baseline on data not used to tune it. Renaming the current geometric pipeline as a VAM would be inaccurate.
