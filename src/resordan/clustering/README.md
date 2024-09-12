# clustering algorithm

This submodule implements the `snr_peaks_detection` function, which is designed to detect events (objects) in radar data using a clustering algorithm based on Signal-to-Noise Ratio (SNR) peaks. It uses polynomial fitting and a configurable loss function to analyze time series of range, range rate, and acceleration, to identify potential targets. The script splits the data into windows based on temporal gaps between SNR peaks, and evaluates each window to determine whether it contains a valid detection. The detected events are then stored in an `EventsDataset`.


## Brief overview of algorithm

1. **Threshold SNR**: Identify data points where SNR exceeds the configured threshold.
1. **Split into Windows**: Group detected SNR peaks into time windows based on gaps in the timestamps.
1. **Polynomial Fitting**: Fit polynomials to range, range rate, and acceleration data in each window.
1. **Compute Loss**: Calculate the difference in polynomial coefficients to evaluate range rate and acceleration consistency.
1. **Threshold Loss**: Compare the computed loss to a threshold to decide if the window contains a detected event.
1. **Create Event**: If detected, store the event's data in an `EventsDataset`.
1. **Return Events**: Return all detected events as the output.


## Example usage

```python
import glob
import logging

from resordan.clustering import algorithm as clustering
from resordan.data.gmf import GMFDataset

logger = logging.getLogger("resordan")
logging.basicConfig(level=logging.INFO)

# Load GMF dataset (change the path to the directory containing your GMF files)
gmf_dir = "../../data/gmf/leo_bpark_2.1u_NO-20221116-UHF/2022-11-16T14-00-00"
gmf_files = list(sorted(glob.glob(f"{gmf_dir}/*.h5")))
gmf_dataset = GMFDataset.from_files(gmf_files)

# Override some parameters for the clustering algorithm
override_detector_params = dict(
    segment_split_time=1.5,
    snr_db_threshold=20,
)
# Run clustering algorithm
events_dataset = clustering.snr_peaks_detection(gmf_dataset, **override_detector_params)

print("Number of detected events:", len(events_dataset.events))
```

## Data format

The algorithm uses the data format defined in the `resordan.data` submodule.

- **Input data:** A `GMFDataset` generated from a collection of GMF output files.
- **Output data:** An `EventsDataset` containing the detected events, and metadata from the input `GMFDataset`.

## In-depth description of algorithm

### Function Signature
```python
def snr_peaks_detection(gmf_dataset: GMFDataset, **kwargs) -> EventsDataset:
```
- **Input:**
  - `gmf_dataset`: An instance of `GMFDataset`, which contains data relevant to the detection process. This includes arrays like time, range, range rate, acceleration, and SNR, all of which are used to detect events.
  - `**kwargs`: Optional keyword arguments that can be passed to override the default values in the `DetectorConfig`.

- **Output:**
  - The function returns an `EventsDataset` object, which contains the detected events that satisfy the detection criteria. Each event includes attributes such as duration, epoch, time, range, range rate, acceleration, SNR, etc.

### Detector config

```python
@dataclass
class DetectorConfig:
    ...
```

Parameters:

- `segment_split_time: float = 5` - Minimum time (in seconds) to consider SNR peaks as beloning to different objects. Default = 5
- `snr_db_threshold: float = 15` - Threshold (in dB) for detecting peaks in SNR. Default = 15
- `loss_weights: tuple[float, float] = (1e-3, 1e-3)` - Loss function weights. Default = (1e-3, 1e-3)
- `loss_threshold: float = 10` - Loss function threshold. Default = 10
- `min_n_samples: int = 5` - Minimum number of samples in an event. Default = 5

### Step-by-Step Description


1. **Thresholding SNR Values**:
    - The function identifies timesteps (indices) where the SNR is greater than a specified threshold (`snr_db_threshold`). This forms the basis for detection, as the algorithm looks for high-SNR peaks to identify potential events.
    - If fewer than four timesteps meet this threshold, the function logs a warning that no detections were found and returns an empty list.

1. **Splitting the Data into Windows**:
    - If enough detections are found, the function proceeds by splitting the detected SNR peaks into separate time windows based on the `segment_split_time` parameter. This parameter defines the minimum time separation required to consider peaks as belonging to different events.

1. **Detection Within Each Window**:
    - For each window (which corresponds to a continuous sequence of detected SNR peaks), the function applies polynomial fitting to the data:
        - It fits a second-degree polynomial to the range data.
        - It fits a first-degree polynomial to the range rate data.
        - It fits a constant (zero-degree) polynomial to the acceleration data.

    - The script calculates two types of loss based on differences between the polynomial coefficients:
        1. **Range Rate Loss**: Measures the difference between the first-degree coefficient of the range polynomial and the constant coefficient of the range rate polynomial. This checks how well the rate of change of the range aligns with the range rate data.
        1. **Acceleration Loss**: Measures the differences between the second-degree coefficient of the range polynomial, the first-degree coefficient of the range rate polynomial, and the constant acceleration polynomial coefficient. This checks the consistency of the acceleration derived from both range and range rate data.

    - The total loss is a weighted sum of the two terms above. The weights for this sum are given by the `loss_weights` tuple.

    - If the total loss is below a certain threshold (`loss_threshold`), the window is classified as a detected event.

1. **Event Creation**:
    - When a window is detected as an event, the function creates an `EventDataset` object for it. This object includes information such as the event duration, the starting time (epoch), and other relevant data like range, range rate, acceleration, and SNR.

1. **Return Detected Events**:
    - After processing all windows, the function returns an `EventsDataset` object, which encapsulates metadata, configuration parameters, and the list of detected events.
