# Car make/model recognition

The application uses the `twincar-group2/twincar-classifier` EfficientNet-B3 checkpoint after YOLO finds a car. It is lightweight enough for local use and predicts 196 fine-grained Stanford Cars classes containing make, model/body style, and an associated model year.

## Stabilization used in this project

- Only YOLO detections classified as `car` are sent to the model.
- Very small crops are skipped.
- At most three spaced-out samples are classified per track.
- Predictions are aggregated by ByteTrack ID.
- A label is displayed only after at least two frames agree and the average confidence exceeds the selected threshold.
- Otherwise the report says `Uncertain`.

## Why confident predictions can still be wrong

Neural-network confidence is not the probability that a real-world answer is correct. The classifier is closed-set: it must assign every crop to one of its 196 known categories. A vehicle absent from those categories can resemble a known class and receive high softmax confidence.

The model's own documentation reports strong results on Stanford Cars-style images but a large domain drop on CompCars. CCTV traffic footage contains smaller cars, different countries/models, occlusion, motion blur, and viewpoints not represented by the training benchmark.

Model card: [twincar-group2/twincar-classifier](https://huggingface.co/twincar-group2/twincar-classifier)

## Improve it for a real deployment

1. Collect and label cropped vehicles from the target camera and location.
2. Define the exact make/model taxonomy you need.
3. Add an `Unknown / other` class and hard negative examples.
4. Fine-tune EfficientNet or MobileNet on the target data.
5. Split training and testing by vehicle identity and camera, not by random near-duplicate frames.
6. Calibrate confidence on a held-out target-domain set.
7. Evaluate make, model, unknown-rejection, and per-class precision/recall separately.

For production automatic number-plate recognition or enforcement, check applicable privacy and transport regulations before collecting or retaining data.
