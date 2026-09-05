#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_PORT I2S_NUM_0

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

    // I2S configuration
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

        .dma_buf_count = 4,

        .dma_buf_len = 128,

        .use_apll = false,

        .tx_desc_auto_clear = false,

        .fixed_mclk = 0
    };

    // GPIO configuration
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN
    };

    // Install driver
    esp_err_t result = i2s_driver_install(
        I2S_PORT,
        &i2s_config,
        0,
        NULL
    );

    if (result != ESP_OK) {
        Serial.println("ERROR: I2S driver installation failed!");
        Serial.print("Error code: ");
        Serial.println(result);
        while (true) {
            delay(1000);
        }
    }

    // Configure pins
    result = i2s_set_pin(
        I2S_PORT,
        &pin_config
    );

    if (result != ESP_OK) {
        Serial.println("ERROR: I2S pin configuration failed!");
        Serial.print("Error code: ");
        Serial.println(result);
        while (true) {
            delay(1000);
        }
    }

    // Clear DMA buffer
    i2s_zero_dma_buffer(I2S_PORT);

    Serial.println("I2S initialized successfully.");
    Serial.println("Listening...");
}

void loop() {

    int32_t samples[128];

    size_t bytes_read = 0;

    esp_err_t result = i2s_read(
        I2S_PORT,
        samples,
        sizeof(samples),
        &bytes_read,
        portMAX_DELAY
    );

    if (result != ESP_OK) {
        Serial.print("I2S read error: ");
        Serial.println(result);
        return;
    }

    size_t samples_read =
        bytes_read / sizeof(int32_t);

    if (samples_read == 0) {
        return;
    }

    int32_t min_sample = INT32_MAX;
    int32_t max_sample = INT32_MIN;

    int64_t sum_abs = 0;

    for (size_t i = 0; i < samples_read; i++) {

        int32_t sample = samples[i];

        if (sample < min_sample) {
            min_sample = sample;
        }

        if (sample > max_sample) {
            max_sample = sample;
        }

        sum_abs += abs((int64_t)sample);
    }

    int32_t average_abs =
        sum_abs / samples_read;

    Serial.print("Samples: ");
    Serial.print(samples_read);

    Serial.print(" | Min: ");
    Serial.print(min_sample);

    Serial.print(" | Max: ");
    Serial.print(max_sample);

    Serial.print(" | AvgAbs: ");
    Serial.println(average_abs);
}