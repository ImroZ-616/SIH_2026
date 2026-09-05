#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 16000

#define I2S_WS   4
#define I2S_SD   6
#define I2S_SCK  5

int32_t rawBuffer[BUFFER_SIZE];
int16_t audioBuffer[BUFFER_SIZE];

void setupI2S()
{
    i2s_config_t config = {
        .mode = (i2s_mode_t)(
            I2S_MODE_MASTER |
            I2S_MODE_RX
        ),

        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format = I2S_COMM_FORMAT_STAND_I2S,

        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 8,
        .dma_buf_len = 256,

        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
}

void setup()
{
    Serial.begin(115200);

    setupI2S();

    Serial.println("Microphone ready!");
}

void loop()
{
    size_t bytesRead;

    i2s_read(
        I2S_PORT,
        rawBuffer,
        sizeof(rawBuffer),
        &bytesRead,
        portMAX_DELAY
    );

    int samplesRead = bytesRead / sizeof(int32_t);

    for (int i = 0; i < samplesRead; i++)
    {
        audioBuffer[i] = rawBuffer[i] >> 14;
    }

    Serial.print("Samples captured: ");
    Serial.println(samplesRead);

    Serial.print("First samples: ");

    for (int i = 0; i < 10 && i < samplesRead; i++)
    {
        Serial.print(audioBuffer[i]);
        Serial.print(" ");
    }

    Serial.println();
}








