#include <Arduino.h>
#include "driver/i2s.h"
#include <stdint.h>

#define I2S_PORT I2S_NUM_0

// Existing microphone wiring from the repository
#define I2S_BCLK 18
#define I2S_WS   23
#define I2S_DIN  35

// EdgeWake audio standard
#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)

// INMP441 commonly outputs 24/32-bit I2S data.
// We read 32-bit samples and convert them to 16-bit PCM.
#define DMA_BUF_COUNT 4
#define DMA_BUF_LEN   128

int16_t audio_buffer[TOTAL_SAMPLES];

void setupI2S() {

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(
            I2S_MODE_MASTER |
            I2S_MODE_RX
        ),

        .sample_rate = SAMPLE_RATE,

        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format = I2S_COMM_FORMAT_STAND_I2S,

        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,

        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN
    };

    esp_err_t result;

    result = i2s_driver_install(
        I2S_PORT,
        &i2s_config,
        0,
        NULL
    );

    if (result != ESP_OK) {
        Serial.println("ERROR: I2S driver installation failed.");
        Serial.println(result);

        while (true) {
            delay(1000);
        }
    }

    result = i2s_set_pin(
        I2S_PORT,
        &pin_config
    );

    if (result != ESP_OK) {
        Serial.println("ERROR: I2S pin configuration failed.");
        Serial.println(result);

        while (true) {
            delay(1000);
        }
    }

    i2s_zero_dma_buffer(I2S_PORT);

    Serial.println("I2S initialized successfully.");
}


void recordAudio() {

    int32_t raw_samples[DMA_BUF_LEN];

    size_t bytes_read = 0;

    size_t total_recorded = 0;

    Serial.println();
    Serial.println("================================");
    Serial.println("       RECORDING AUDIO");
    Serial.println("================================");

    Serial.println("Sample rate : 16000 Hz");
    Serial.println("Duration    : 1 second");
    Serial.println("Target      : 16000 samples");

    unsigned long start_time = millis();

    while (total_recorded < TOTAL_SAMPLES) {

        esp_err_t result = i2s_read(
            I2S_PORT,
            raw_samples,
            sizeof(raw_samples),
            &bytes_read,
            portMAX_DELAY
        );

        if (result != ESP_OK) {
            Serial.println("ERROR: I2S read failed.");
            return;
        }

        size_t samples_read =
            bytes_read / sizeof(int32_t);

        for (size_t i = 0;
             i < samples_read && total_recorded < TOTAL_SAMPLES;
             i++) {

            /*
             * INMP441 data arrives in a 32-bit I2S container.
             *
             * Convert to approximately 16-bit PCM.
             *
             * The exact shift can depend on microphone/data format,
             * so this is intentionally kept simple for the first
             * hardware validation stage.
             */
            audio_buffer[total_recorded] =
                (int16_t)(raw_samples[i] >> 14);

            total_recorded++;
        }
    }

    unsigned long elapsed =
        millis() - start_time;

    Serial.println();
    Serial.println("Recording complete.");

    Serial.print("Samples captured : ");
    Serial.println(total_recorded);

    Serial.print("Expected samples : ");
    Serial.println(TOTAL_SAMPLES);

    Serial.print("Capture time     : ");
    Serial.print(elapsed);
    Serial.println(" ms");


    // Calculate statistics

    int16_t min_sample = INT16_MAX;
    int16_t max_sample = INT16_MIN;

    int64_t sum_abs = 0;

    int zero_count = 0;

    for (size_t i = 0;
         i < TOTAL_SAMPLES;
         i++) {

        int16_t sample =
            audio_buffer[i];

        if (sample < min_sample)
            min_sample = sample;

        if (sample > max_sample)
            max_sample = sample;

        if (sample == 0)
            zero_count++;

        sum_abs +=
            abs((int32_t)sample);
    }

    float average_abs =
        (float)sum_abs / TOTAL_SAMPLES;


    Serial.println();
    Serial.println("========== AUDIO STATS ==========");

    Serial.print("Min sample      : ");
    Serial.println(min_sample);

    Serial.print("Max sample      : ");
    Serial.println(max_sample);

    Serial.print("Average abs     : ");
    Serial.println(average_abs);

    Serial.print("Zero samples    : ");
    Serial.println(zero_count);

    Serial.println("=================================");


    // Basic validation

    if (total_recorded == TOTAL_SAMPLES) {

        Serial.println();
        Serial.println("PASS: 16000 samples captured.");

    } else {

        Serial.println();
        Serial.println("FAIL: Incorrect sample count.");
    }


    if (average_abs < 10) {

        Serial.println(
            "WARNING: Audio level is very low."
        );

    } else {

        Serial.println(
            "PASS: Microphone is producing signal."
        );
    }


    Serial.println();
    Serial.println(
        "Audio buffer is ready for MFCC."
    );
}


void setup() {

    Serial.begin(115200);

    delay(1000);

    Serial.println();
    Serial.println("================================");
    Serial.println("      EDGEWAKE MIC TEST");
    Serial.println("================================");

    Serial.println();
    Serial.println("Initializing I2S...");

    setupI2S();

    delay(1000);
}


void loop() {

    recordAudio();

    Serial.println();
    Serial.println(
        "Waiting 3 seconds before next capture..."
    );

    delay(3000);
}
