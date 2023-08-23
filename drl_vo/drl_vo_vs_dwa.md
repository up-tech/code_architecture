### DRL_VO VS DWA in simulation

#### experimental method

Separately using DRL_VO and DWA finish a sequence of navigation tasks. statistic and compare their total time spent, trajectory length, success ratel.

#### example of result on console

```
[INFO] [1692773895.010928, 392.831000]: Goal pose 25 reached
[INFO] [1692773895.011968, 392.833000]: Start Time: 36.497 secs
[INFO] [1692773895.012526, 392.833000]: End Time: 392.833 secs
[INFO] [1692773895.013119, 392.834000]: Success Number: 25 in total number 25
[INFO] [1692773895.013651, 392.834000]: Total Running Time: 356.336 secs
[INFO] [1692773895.014304, 392.835000]: Total Trajectory Length: 136.6251028026126 m
[INFO] [1692773895.014934, 392.836000]: Final goal pose reached!
```

---

#### autolab_without_pedestrians

<img src="image/autolab.png" style="zoom:15%;" />

| Environment | Method | Success rate | Average time (s) | Average length (m) | Average speed (m/s) |
| ----------- | ------ | ------------ | ---------------- | ------------------ | ------------------- |
| autolab     | DRL_VO | 1.0          | 12.49            | 5.88               | 0.47                |
| autolab     | DWA    | 1.0          | 14.25            | 5.47               | 0.38                |

#### autolab_with_pedestrians

<img src="image/autolab_25ped.png" style="zoom:20%;" />

| Environment   | Method | Success rate | Average time (s) | Average length (m) | Average speed (m/s) |
| ------------- | ------ | ------------ | ---------------- | ------------------ | ------------------- |
| autolab_25ped | DRL_VO | 0.76         | 12.94            | 5.92               | 0.46                |
| autolab_25ped | DWA    | 0.96         | 18.07            | 5.53               | 0.31                |