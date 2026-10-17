# Hand FSR Sensor Acquisition and Visualization

A Python-based sensor acquisition and visualization application developed as a companion module for an upper-limb rehabilitation robot project.

The application processes data from a hand force-sensing array and presents the measurements through a real-time graphical interface. It is designed to support sensor monitoring, visualization, and further development of data-driven hand-function assessment workflows.

## Project Context

This repository contains the sensor acquisition and visualization module associated with the [Upper-Limb Rehabilitation Robot — ROS 2](https://github.com/Ruoyizhu/upper-limb-rehabilitation-robot-ros2) project.

The sensor application is maintained separately to keep the data acquisition and visualization logic modular and easier to develop and test.

**Current integration boundary:** The application includes standalone serial acquisition and visualization functionality. ROS 2 communication with the main rehabilitation robot is a separate integration task and is not claimed as a completed feature.

## Key Features

### 1. Sensor Data Acquisition

- Serial-port discovery and connection management.
- Configurable serial baud rate.
- Incremental text-stream decoding and frame parsing.
- Reconstruction of sensor measurements into a 16 × 16 matrix.
- Handling of incomplete data, malformed rows, and stream resynchronization.
- Monitoring of incoming data volume and acquisition status.

### 2. Real-Time Heatmap Visualization

- A 16 × 16 grid representing 256 sensing locations.
- Adaptive layout with square cells.
- Adjustable color range and cell spacing.
- Color mapping for visualizing the spatial distribution of sensor values.
- A zeroing function to establish a reference measurement.

### 3. Time-Series Analysis

The graphical interface displays measurement statistics and time-series plots, including:

- Total force.
- Current peak value.
- Session maximum and minimum values.
- Incoming data rate and UI refresh rate.
- Selectable plotting time windows.

The displayed physical units and measurement accuracy depend on the sensor characteristics and the validity of the calibration procedure. The GUI's displayed values should not be interpreted as clinically validated force measurements without independent calibration.

### 4. Responsive GUI Architecture

The application uses PyQt6 and Qt's threading and signal mechanisms to separate serial acquisition from interface updates.

- Background acquisition runs in a worker thread.
- Qt signals communicate data and status changes to the GUI.
- UI refresh is independently controlled by a timer.
- Recent frames are cached to limit unnecessary rendering and reduce event backlog.
- Display settings can be persisted through `QSettings`.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Application logic |
| PyQt6 | Desktop GUI and thread management |
| NumPy | Sensor matrix processing and statistics |
| pyqtgraph | Real-time plotting |
| pyserial | Serial communication with the sensor device |

## Architecture

```text
Hand FSR Sensor Array
        |
        v
  Serial Interface
        |
        v
  SerialByteSource
        |
        v
  TextFrameParser
        |
        v
  16 x 16 Sensor Matrix
        |
        +----------------------+
        |                      |
        v                      v
  Heatmap Display       Statistical Analysis
                               |
                               v
                        Time-Series Plots

Acquisition Worker Thread
        |
        | Qt signals
        v
     GUI Thread
```

The architecture separates the input stream, frame parsing, data representation, and presentation layer. This makes it easier to debug serial communication independently from visualization.

## Repository Structure

```text
hand-fsr-sensor-visualization/
├── src/
│   └── main.py                 # GUI, serial acquisition and visualization
├── requirements.txt            # Python dependencies
├── hand-fsr-demo-progress.md   # Development notes
├── LICENSE
└── README.md
```

The repository may contain additional development artifacts that are not required for a clean installation.

## Prerequisites

- Python 3.10 or later.
- A compatible desktop environment.
- A serial-connected sensor device for hardware acquisition.
- A suitable serial driver and access permissions.

The GUI and plotting components can be developed independently of the physical sensor. Hardware acquisition requires a device that follows the expected input format.

## Installation

### Windows

Open PowerShell in the repository directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, you can run the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the system does not provide the required Python or Qt dependencies, install the appropriate packages for your operating system before proceeding.

## Dependency File

The current source code imports `pyqtgraph` and optionally uses `pyserial`, so make sure `requirements.txt` includes all required packages:

```text
PyQt6>=6.7
numpy>=1.26
pyqtgraph>=0.13
pyserial>=3.5
```

After updating the file, install the dependencies again.

## Running the Application

From the repository root:

### Windows

```powershell
.\.venv\Scripts\python.exe src\main.py
```

### Linux

```bash
python src/main.py
```

The application opens a desktop window with sensor visualization, data statistics, and serial controls.

## Using the Application

1. Launch the application.
2. Connect the sensor hardware using the appropriate serial interface.
3. Refresh the available serial ports.
4. Select the port and baud rate matching the sensor device.
5. Open the serial connection.
6. Observe the sensor heatmap and time-series plots as data arrives.
7. Use the zeroing control and display settings as appropriate for the measurement setup.
8. Close the serial connection before disconnecting the device.

The device's communication protocol, sampling configuration, and physical calibration must match the actual hardware. The default baud rate and display scaling are not universal sensor specifications.

## Data Processing Details

The current parser supports a text-based serial stream containing rows of numeric values. It handles incremental byte input and reconstructs complete frames from the incoming rows.

The resulting frame is represented as a NumPy array with shape `(16, 16)`.

The application also tracks parser statistics, including accepted rows, dropped lines, emitted frames, and resynchronization events. These metrics can help diagnose communication problems and malformed input.

## Validation

Useful validation steps include:

- Checking that the application starts with the documented dependencies.
- Testing serial connection and disconnection.
- Verifying the reconstruction of known 16 × 16 input frames.
- Testing incomplete and malformed input lines.
- Confirming that the UI remains responsive during data acquisition.
- Comparing displayed measurements against a known reference before using them for quantitative analysis.

The validity of force measurements depends on the physical sensor, its calibration, and the interpretation of its output.

## Current Status and Future Work

The current code provides a standalone acquisition and visualization application with serial input, matrix reconstruction, heatmap rendering, time-series plotting, and GUI controls.

Potential next steps include:

- Documenting the hardware protocol and sensor calibration procedure.
- Adding reproducible data recording and replay.
- Adding automated parser and GUI tests.
- Integrating the sensor stream into the rehabilitation robot's ROS 2 architecture.
- Defining and validating a standard sensor message interface.

These items are future development directions unless explicitly implemented and tested.

## Related Project

- [Upper-Limb Rehabilitation Robot — ROS 2](https://github.com/Ruoyizhu/upper-limb-rehabilitation-robot-ros2)

## License

See [LICENSE](LICENSE) for the applicable license and terms.
