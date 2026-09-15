# Custom trackers

Pass a configured tracker instance to `model.track()` to use an algorithm
outside LibreYOLO without forking or registering it. The built-in tracker
names continue to work.

```python
from libreyolo import LibreYOLO
from libreyolo.tracking import ByteTracker


class MyTracker:
    # Replace this delegate with your own association implementation.
    def __init__(self):
        self.delegate = ByteTracker(track_low_thresh=0.1)

    def reset(self):
        self.delegate.reset()

    def update(self, results, image=None):
        return self.delegate.update(results, image=image)


model = LibreYOLO("LibreYOLO9s.pt")
for result in model.track("video.mp4", tracker=MyTracker(), track_conf=0.1):
    print(result.boxes.id)
```

The same entry point is inherited by RF-DETR and other models using the base
tracking implementation. Existing restrictions on tasks and sources still
apply. Video files, image directories, lists, and lazy image iterators use
the same custom tracker interface.

## Interface and lifetime

`libreyolo.tracking.Tracker` is a typing protocol. Inheriting from it is
optional; an instance needs callable `reset()` and `update(results,
image=None)` methods.

- `reset()` runs once when the tracking generator starts processing a valid
  sequence. Reusing the instance in another call starts a fresh sequence.
  An empty image directory returns without resetting or updating it.
- `update()` receives detection `Results` and the original RGB PIL image as
  the `image` keyword, once per retained frame, including empty detections.
- Return `Results` with a one-dimensional integer tensor/array in `track_id`,
  one ID per returned box. Empty output needs an empty ID vector. Use the
  boxes' backend and device. LibreYOLO also exposes these IDs as `boxes.id`.
- Use `results[indices]` when filtering/reordering detections so masks and
  keypoints follow the selected boxes. Their alignment is the adapter's
  responsibility.
- Configure the instance before passing it. `tracker_config` and additional
  tracker keyword arguments are rejected with an instance.
- For an instance, `track_conf` is the **detector** confidence cutoff (default
  `0.25`). Set it low enough to retain weak detections needed by your tracker.
  LibreYOLO does not inspect or modify the instance's association thresholds
  or frame-rate settings. Configure timing for the retained frame rate,
  accounting for `vid_stride` yourself.
- Give each concurrent run/camera its own instance. To retain state across
  manually managed batches, call the tracker directly after `model.predict()`;
  separate `model.track()` calls reset state.

A custom tracker can wrap a Python algorithm or native bindings. This API
makes no changes to DeepStream integration and makes no performance claims.
