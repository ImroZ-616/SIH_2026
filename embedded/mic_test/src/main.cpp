#include <Arduino.h>
#include <ESP_I2S.h>

I2SClass I2S;

// INMP441 → ESP32-S3
#define I2S_BCLK 18
#define I2S_WS   23
#define I2S_DIN  35

#define SAMPLE_RATE 16000

void setup() {
    Serial.begin(115200);

    delay(1000);

    Serial.println();
    Serial.println("================================");
    Serial.println(" ASTRAEDGE INMP441 MIC TEST");
    Serial.println("================================");

    // Configure I2S pins
    I2S.setPins(
        I2S_BCLK,   // BCLK / SCK
        I2S_WS,     // WS / LRCLK
        I2S_DIN,    // DATA IN
        -1,         // DATA OUT unused
        -1          // MCLK unused
    );

    // Configure I2S
    if (!I2S.begin(
        I2S_MODE_STD,
        SAMPLE_RATE,
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_MONO,
        I2S_STD_SLOT_LEFT
    )) {
        Serial.println("ERROR: I2S initialization failed!");
        while (true) {
            delay(1000);
        }
    }

    Serial.println("I2S initialized successfully.");
    Serial.println("Listening...");
}

void loop() {
    int32_t samples[128];

    size_t bytesRead = I2S.readBytes(
        (char *)samples,
        sizeof(samples)
    );

    size_t samplesRead = bytesRead / sizeof(int32_t);

    if (samplesRead == 0) {
        Serial.println("No audio data received.");
        delay(100);
        return;
    }

    int32_t minSample = INT32_MAX;
    int32_t maxSample = INT32_MIN;
    int64_t sumAbs = 0;

    for (size_t i = 0; i < samplesRead; i++) {
        int32_t sample = samples[i];

        if (sample < minSample) {
            minSample = sample;
        }

        if (sample > maxSample) {
            maxSample = sample;
        }

        sumAbs += abs((long)sample);
    }

    int32_t averageAbs = sumAbs / samplesRead;

    Serial.print("Samples: ");
    Serial.print(samplesRead);

    Serial.print(" | Min: ");
    Serial.print(minSample);

    Serial.print(" | Max: ");
    Serial.print(maxSample);

    Serial.print(" | AvgAbs: ");
    Serial.println(averageAbs);
}