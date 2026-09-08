# System Architecture

**Project:** ASTRAEDGE
**System:** Low-Latency and Efficient Voice Activator for Edge Devices
**Owner:** R6 — Integration/System Engineer

---

## 1. System Objective

ASTRAEDGE uses a hybrid edge-cloud architecture.

The edge device continuously receives microphone audio and performs local Keyword Spotting (KWS). Audio remains local while the system is listening.

When the target wake word is detected, the system transitions from local listening to active speech streaming. Only the required audio is sent to the remote server for Automatic Speech Recognition (ASR).

The architecture is designed to minimize:

- Wake-up latency
- False activations
- Continuous network bandwidth usage
- Edge memory usage
- Unnecessary cloud processing

---

## 2. High-Level Architecture

The system consists of two major parts:

**Edge side**

Microphone → Audio Capture → Preprocessing → Feature Extraction → Local KWS → WakeController → Ring Buffer → WebSocket Client

**Cloud/server side**

WebSocket Server → Audio Receiver → ASR Engine → Transcription → Response

The edge performs the latency-sensitive wake-word detection locally, while the server performs computationally expensive speech recognition.

---

## 3. Design Principle

The central architectural principle is:

**Detect the wake word locally and stream only the required speech.**

This avoids continuously transmitting microphone audio to the cloud.

The architecture therefore separates responsibilities:

- Edge: low-latency wake detection and local audio handling
- Network: transfer audio only after activation
- Server: high-compute speech recognition

---

## 4. System Boundary

The edge system is responsible for:

- Capturing microphone audio
- Maintaining the recent audio buffer
- Running the KWS detector
- Detecting the wake event
- Starting and stopping audio streaming
- Handling local integration state

The remote server is responsible for:

- Receiving streamed audio
- Managing the audio session
- Running ASR
- Returning transcription or errors

The network boundary exists between the WebSocket client on the edge and the WebSocket server.---

## 5. Component Responsibilities

### 5.1 Audio Capture

The Audio Capture layer receives microphone samples from the input device.

Responsibilities:

- Capture audio continuously
- Maintain the configured sample rate
- Produce fixed-size audio chunks
- Provide audio samples to the processing pipeline

Baseline configuration:

- Sample rate: 16 kHz
- Channels: Mono
- Sample format: 16-bit PCM

---

### 5.2 Preprocessing and Feature Extraction

The preprocessing layer prepares microphone audio for KWS inference.

Responsibilities:

- Normalize audio format
- Maintain the required sampling rate
- Convert audio into the representation expected by the KWS model
- Provide feature frames to the detector

The current KWS pipeline uses MFCC-based features.

---

### 5.3 Local KWS Detector

The KWS detector determines whether the target wake word is present.

The detector follows the KWS interface:

- `process(audio_chunk)` → detection result
- `reset()` → reset detector state

The implementation is replaceable so that the current mock detector can later be replaced by the trained TinyML KWS model.

---

### 5.4 WakeController

WakeController is the integration component responsible for connecting KWS detection with audio streaming.

Responsibilities:

- Receive incoming audio
- Maintain the recent audio history
- Process KWS results
- Trigger the wake event
- Switch the system into streaming mode
- Return buffered audio when the wake word is detected
- Stop streaming when the interaction ends

This component acts as the main state-transition boundary between local KWS and network streaming.

---

### 5.5 Audio Ring Buffer

The ring buffer stores the most recent microphone samples while the system is listening.

Its purpose is to preserve a short period of audio immediately before wake-word detection.

When the wake word is detected:

**Ring Buffer → Buffered Audio → WebSocket Streaming**

This reduces the risk of losing the beginning of the user's command because wake detection occurs after the command has already started.

---

### 5.6 WebSocket Client

The WebSocket client provides the network interface between the edge system and the ASR server.

The current protocol is:

**START → AUDIO → END → TRANSCRIPTION**

Responsibilities:

- Open the WebSocket connection
- Start an audio session
- Send audio chunks
- Send the buffered audio after wake detection
- Send subsequent microphone audio
- Signal the end of the interaction
- Receive transcription or error responses---

## 6. System State Machine

ASTRAEDGE operates as a state-driven system.

The primary interaction flow is:

**LISTEN → WAKE → STREAM → ASR → STOP**

### 6.1 LISTEN

Initial state of the system.

The microphone is active and audio is continuously processed locally.

Actions:

- Capture microphone audio
- Update the ring buffer
- Run the local KWS detector
- Do not transmit microphone audio to the server

The system remains in this state until the target wake word is detected.

---

### 6.2 WAKE

The WAKE state is entered when the KWS detector reports a positive detection.

Actions:

- Confirm the wake event
- Retrieve buffered audio
- Switch the WakeController to streaming mode
- Prepare the network session

The buffered audio is preserved so that speech immediately preceding the wake detection is not lost.

---

### 6.3 STREAM

The system begins transmitting audio to the remote server.

Actions:

- Send the START message
- Send buffered audio
- Continue sending new microphone audio chunks
- Maintain the active WebSocket session

The edge device does not perform full speech recognition locally.

---

### 6.4 ASR

The remote server receives the audio and performs Automatic Speech Recognition.

Actions:

- Receive audio data
- Assemble the audio session
- Run the ASR engine
- Generate the transcription
- Return the transcription to the edge client

The current server implementation performs batch transcription after the END message.

---

### 6.5 STOP

The interaction ends when the user command has been processed.

Actions:

- Send the END message
- Receive the transcription or error
- Stop active streaming
- Reset the controller state
- Prepare the system for the next interaction

The system then returns to:

**STOP → LISTEN**

---

### 6.6 State Transition Summary

| Current State | Event | Next State |
|---|---|---|
| LISTEN | Wake word detected | WAKE |
| WAKE | Buffered audio prepared | STREAM |
| STREAM | Audio transmission active | ASR |
| ASR | Transcription received | STOP |
| STOP | Session reset | LISTEN |---

## 7. End-to-End Data Flow

The complete data path is:

**Microphone → Audio Capture → Preprocessing → KWS → WakeController → Ring Buffer → WebSocket → ASR → Transcription**

### 7.1 Listening Path

During normal operation:

1. Microphone produces PCM audio samples.
2. Audio Capture divides the samples into processing chunks.
3. Preprocessing prepares the audio for KWS.
4. The KWS detector processes each chunk.
5. WakeController stores recent samples in the ring buffer.
6. No audio is transmitted to the server.

This keeps normal listening local to the edge device.

---

### 7.2 Wake Detection Path

When the KWS detector detects the target keyword:

1. KWS returns a positive detection.
2. WakeController receives the detection event.
3. WakeController changes the system to streaming mode.
4. The ring buffer returns the most recent buffered audio.
5. The WebSocket client starts the remote audio session.
6. Buffered audio is transmitted first.

This creates the bridge between local wake detection and remote speech recognition.

---

### 7.3 Active Streaming Path

After the wake event:

1. New microphone samples continue arriving.
2. Audio chunks are passed to the WakeController.
3. WakeController forwards new samples while streaming is active.
4. The WebSocket client sends the audio to the server.
5. The server receives and stores the audio.
6. The ASR engine processes the completed audio session.

---

### 7.4 Response Path

After ASR processing:

1. The server generates a transcription.
2. The server sends a transcription message to the client.
3. The client receives the response.
4. The streaming session is terminated.
5. WakeController returns to the non-streaming state.
6. The system resumes local listening.

---

## 8. Interface Boundaries

The architecture uses explicit interfaces between major components.

### 8.1 Audio Interface

Input:

- Microphone PCM samples

Format:

- 16 kHz
- Mono
- 16-bit signed PCM

Output:

- Fixed-size audio chunks

---

### 8.2 KWS Interface

The KWS detector exposes:

- `process(audio_chunk)` → Boolean detection result
- `reset()` → reset detector state

The interface allows different KWS implementations to be used without changing the rest of the integration pipeline.

The current integration uses MockKWS for system testing.

The trained KWS implementation will replace the mock implementation when model integration is completed.

---

### 8.3 WakeController Interface

WakeController receives:

- Audio chunks
- KWS detection results

WakeController produces:

- Buffered audio after wake detection
- Live audio chunks while streaming

The controller also maintains the streaming state.

---

### 8.4 WebSocket Interface

The network interface uses a session-based message protocol:

**START**

Creates a new audio recognition session.

**AUDIO**

Transfers raw PCM audio bytes.

**END**

Signals that the current audio session is complete.

**TRANSCRIPTION**

Returns the recognized speech from the ASR server.

**ERROR**

Reports a server-side or session-level failure.

---

## 9. Current Integration Boundary

The current R6 implementation integrates:

- Audio chunk generation
- Ring buffer
- WakeController
- KWS interface
- Mock KWS detector
- WebSocket client
- WebSocket server
- ASR server
- End-to-end integration tests

The KWS model is currently represented by a mock detector for integration testing.

The trained model, embedded deployment, and hardware optimization remain separate component responsibilities.

This separation allows each subsystem to be developed and tested independently before final hardware integration.---

## 10. Error Handling and Failure Paths

The system must remain stable when individual components fail.

Failures should be isolated so that a temporary microphone, network, or ASR problem does not permanently stop the system.

### 10.1 Microphone Failure

If microphone input becomes unavailable:

- Stop processing invalid audio
- Report the capture error
- Prevent corrupted audio from entering the KWS pipeline
- Attempt recovery when the audio source becomes available

The system should return to the LISTEN state after successful recovery.

---

### 10.2 KWS Failure

If the KWS detector encounters an inference or processing error:

- Reject the invalid inference result
- Keep the current audio pipeline stable
- Record the error for debugging
- Reset the detector state
- Continue local listening when possible

A KWS failure must not automatically start network streaming.

---

### 10.3 False Activation

A false activation occurs when the KWS detector reports the wake word when the user did not intend to activate the system.

The system should:

- Start streaming only after a valid KWS event
- Keep the activation decision local
- Record false activation events during testing
- Measure false activation rate during benchmarking

False activation is a key quality metric for the final system.

---

### 10.4 Missed Activation

A missed activation occurs when the user says the target keyword but the detector does not trigger.

The system should:

- Continue listening
- Continue updating the ring buffer
- Avoid entering the streaming state
- Record the event during evaluation when ground truth is available

Missed activations are evaluated together with detection accuracy and false activation rate.

---

### 10.5 Ring Buffer Overflow

The ring buffer has a fixed maximum capacity.

When the buffer becomes full:

- The oldest samples are overwritten
- The newest samples are retained
- Memory usage remains bounded

This prevents unbounded audio memory growth on the edge device.

---

### 10.6 Network Failure

If the WebSocket connection fails:

- Stop active streaming
- Report the network error
- Close the failed session
- Reset the streaming state
- Return to local listening when recovery is possible

The system must not remain permanently stuck in the STREAM state.

---

### 10.7 ASR Timeout

If the ASR server does not respond within the expected session time:

- Terminate the active recognition session
- Report an ASR timeout
- Reset the controller state
- Return to LISTEN

Timeout handling prevents a failed server request from blocking future wake events.

---

### 10.8 Corrupted or Invalid Audio

If received audio is invalid:

- Reject the affected data
- Do not pass corrupted data to ASR
- Report the protocol or audio error
- Terminate the affected session if recovery is not possible

The next interaction should begin with a clean session.

---

## 11. Recovery Strategy

The general recovery strategy is:

**Detect Failure → Isolate Component → Reset State → Recover → Resume LISTEN**

The integration layer should maintain clear state boundaries so that a failed interaction can be discarded without corrupting the next interaction.

A new interaction must always start from a clean controller and network session state.---

## 12. Performance and Resource Considerations

ASTRAEDGE is designed for resource-constrained edge devices.

The architecture separates lightweight local wake-word detection from computationally expensive cloud ASR.

### 12.1 Latency

The most latency-sensitive path is:

**KWS Detection → WakeController → Buffered Audio → Network Transmission**

The integration layer minimizes processing between wake detection and the beginning of audio transmission.

The current integration tests measure:

- KWS-to-wake handling latency
- Wake-to-first-network-audio latency
- Wake-to-final-transcription latency

These measurements are used as integration baselines.

The current integration uses a mock KWS detector, so final product latency must be measured again after the trained KWS model is deployed.

---

### 12.2 Memory

The ring buffer uses a fixed-size preallocated NumPy array.

For 16 kHz mono 16-bit PCM audio:

- 0.25 seconds = approximately 7.8 KB
- 0.50 seconds = approximately 15.6 KB
- 1.00 second = approximately 31.2 KB
- 1.50 seconds = approximately 46.9 KB
- 2.00 seconds = approximately 62.5 KB

The current baseline uses a 1-second ring buffer.

The buffer size can be adjusted after real hardware measurements determine the required pre-roll duration.

---

### 12.3 Network Bandwidth

During LISTEN:

**Network audio transmission = 0**

Only local KWS processing occurs.

During STREAM:

Audio is transmitted only for the active interaction.

For 16 kHz mono 16-bit PCM:

**Raw audio rate = 32 KB/s approximately**

This architecture therefore reduces unnecessary network traffic compared with continuously uploading microphone audio.

---

### 12.4 CPU Utilization

The edge device performs:

- Audio capture
- Preprocessing
- KWS inference
- Buffer management
- Network transmission during active sessions

The computationally expensive ASR workload remains on the remote server.

CPU utilization must be measured on the target embedded hardware after the trained KWS model is deployed.

---

### 12.5 Flash Usage

The final embedded deployment must account for:

- KWS model size
- Runtime libraries
- Audio processing code
- Network stack
- Application firmware

Model size and firmware size should be measured from the final embedded build.

The architecture does not assume that the current Python implementation represents the final embedded memory footprint.

---

### 12.6 Power

The architecture is intended to reduce unnecessary radio and network activity.

During normal LISTEN operation, the device performs local KWS instead of continuously transmitting audio.

Power consumption should be measured on the target hardware in at least two operating modes:

- Continuous listening
- Active streaming

The measured values should be reported by the hardware benchmarking component.

---

## 13. Integration Metrics

The following metrics should be tracked during system validation:

| Metric | Purpose |
|---|---|
| Wake detection latency | Measures local activation responsiveness |
| Wake-to-first-network-audio | Measures integration handoff speed |
| Wake-to-transcription | Measures overall interaction latency |
| False activation rate | Measures unwanted activations |
| Missed activation rate | Measures detection reliability |
| RAM usage | Measures memory footprint |
| Flash usage | Measures firmware/model footprint |
| CPU utilization | Measures processing load |
| Network bandwidth | Measures communication efficiency |
| Power consumption | Measures energy efficiency |

Hardware-dependent metrics must be measured on the target embedded platform rather than inferred from desktop execution.---

## 14. Testing and Validation Architecture

Testing is performed at multiple levels so that individual components can be validated before complete system integration.

### 14.1 Unit Testing

Unit tests validate individual components independently.

Examples:

- Ring buffer insertion
- Ring buffer wraparound
- Ring buffer clearing
- KWS interface behavior
- WakeController state transitions
- Audio format handling

Unit tests should be deterministic and fast.

---

### 14.2 Integration Testing

Integration tests validate communication between multiple components.

Important integration paths include:

**KWS → WakeController**

**WakeController → Ring Buffer**

**WakeController → WebSocket Client**

**WebSocket Client → WebSocket Server**

**WebSocket Server → ASR**

Integration tests verify that data and state transitions occur correctly across component boundaries.

---

### 14.3 End-to-End Testing

The complete system path is tested as a single workflow:

**Microphone → KWS → Wake → Streaming → ASR → Transcription**

End-to-end tests should verify:

- Wake event generation
- Buffered audio delivery
- Post-wake audio delivery
- Network session creation
- Audio transmission
- ASR execution
- Transcription reception
- Session termination
- Return to LISTEN

---

### 14.4 Failure Testing

Failure scenarios must also be tested.

Examples include:

- Microphone failure
- KWS inference failure
- False activation
- Missed activation
- Network disconnection
- ASR timeout
- Invalid audio
- Buffer overflow
- Repeated start/stop cycles

The expected result is that the system returns to a known state and can process a subsequent interaction.

---

### 14.5 Hardware-in-the-Loop Testing

After embedded deployment, system validation should be repeated on the target hardware.

Hardware testing should measure:

- KWS inference latency
- Wake-to-stream latency
- RAM usage
- Flash usage
- CPU utilization
- Power consumption
- Network performance
- False activation behavior
- Missed activation behavior

Desktop measurements should not be treated as final hardware measurements.

---

## 15. Validation Criteria

The integration is considered functionally complete when:

1. The edge device continuously receives microphone audio.
2. Local KWS can generate a wake event.
3. WakeController transitions into streaming mode.
4. Buffered audio is preserved and transmitted.
5. New microphone audio is streamed after activation.
6. The server receives the complete audio session.
7. ASR produces a transcription.
8. The client receives the transcription or an error.
9. Streaming stops correctly.
10. The controller returns to the LISTEN state.
11. A second interaction can start without restarting the application.

Performance completion additionally requires measured hardware results for the target embedded platform.---

## 16. Component Ownership

ASTRAEDGE is developed as a multi-component system.

Each role owns a specific subsystem while R6 integrates the components into the complete workflow.

| Role | Primary Responsibility |
|---|---|
| R1 | Audio DSP, capture, preprocessing and audio pipeline |
| R2 | KWS dataset, features, model training and accuracy |
| R3 | TinyML model deployment on embedded hardware |
| R4 | Networking and ASR server |
| R5 | Optimization, quantization and hardware benchmarking |
| R6 | System integration, architecture, testing and validation |

R6 does not replace the implementation work owned by the other roles.

Instead, R6 defines the interfaces and verifies that the components operate together correctly.

---

## 17. Integration Dependencies

The final system depends on the following interfaces:

### R1 → R2

Audio preprocessing must provide input in the format expected by the KWS feature pipeline.

### R2 → R3

The trained KWS model must be provided in a format suitable for embedded deployment.

### R3 → R6

The embedded KWS implementation must expose a detection event that can be connected to the system controller.

### R6 → R4

The integration layer sends activated audio through the agreed WebSocket protocol.

### R5 → R6

Hardware benchmarking provides measured resource and latency results for final system validation.

---

## 18. Current Implementation Status

The current integration baseline contains:

- Audio chunk processing
- Fixed-size ring buffer
- WakeController
- KWS detector interface
- Mock KWS implementation
- WebSocket client
- WebSocket server
- ASR integration
- Wake-to-network latency tests
- End-to-end KWS-to-ASR integration tests
- Buffer size measurements
- Integration documentation

The current KWS integration uses MockKWS for deterministic system testing.

The final system will replace the mock detector with the trained and embedded KWS implementation.

---

## 19. Architectural Evolution

The architecture is designed to evolve without changing the complete system.

Current development path:

**Mock KWS → Trained KWS → Quantized KWS → Embedded KWS**

The surrounding integration pipeline remains based on the same logical event:

**KWS Detection → WakeController → Streaming**

This modular design allows model development, embedded optimization, networking, and system integration to proceed independently.

---

## 20. Final System Architecture

The final intended architecture is:

**Microphone**
↓
**Audio Capture**
↓
**Local Preprocessing**
↓
**TinyML KWS**
↓
**WakeController**
↓
**Ring Buffer**
↓
**WebSocket Client**
↓
**Network**
↓
**WebSocket Server**
↓
**ASR Engine**
↓
**Transcription**
↓
**Client**

The critical architectural decision is that the wake word is detected locally before network audio streaming begins.

This provides the foundation for:

- Low activation latency
- Reduced unnecessary bandwidth
- Local wake-word privacy
- Bounded edge memory usage
- Modular KWS replacement
- Independent server-side ASR scaling

The architecture will be considered final only after the trained KWS model and target embedded hardware are integrated and the required performance metrics are measured.
