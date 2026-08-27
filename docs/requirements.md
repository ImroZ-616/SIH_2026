# System Requirements

## Problem Statement

Low Latency and Efficient Voice Activator for Edge Devices.

## Organization

Indian Space Research Organisation (ISRO)

## Core Objective

Build an ultra-lightweight Keyword Spotting (KWS) system that runs locally
on a low-power edge device and streams subsequent audio to a remote ASR
server after detecting a custom keyword.

## Functional Requirements

- Custom keyword detection
- Local/on-device inference
- Continuous low-power listening
- Low false activation rate
- Low detection latency
- Audio buffering
- Audio streaming after wake detection
- Remote ASR integration

## Performance Requirements

The system will be evaluated using:

- Model size
- RAM usage
- Flash usage
- CPU utilization
- Keyword detection accuracy
- False activation rate
- False rejection rate
- KWS inference latency
- Wake-to-ASR latency
- Power consumption

## Restrictions

- Open-source technologies only
- No proprietary voice activation SDKs
- No pre-trained global wake words
- Custom keyword must be trained by the team