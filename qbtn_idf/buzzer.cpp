#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "driver/gpio.h"

#define BUZZER_PIN  (gpio_num_t)25

void gpio_init()
{
    gpio_pad_select_gpio(BUZZER_PIN);
    gpio_set_direction(BUZZER_PIN, GPIO_MODE_OUTPUT);
}

void buzzer(int ms)
{
    int cycles = ms * 4;

    for(int i = 0; i < cycles; i++)
    {
        gpio_set_level(BUZZER_PIN, 1);
        ets_delay_us(250);
        gpio_set_level(BUZZER_PIN, 0);
        ets_delay_us(250);
    }
}

extern "C" void app_main()
{
    gpio_init();

    for (int i = 0; i < 3; i++)
    {
        buzzer(1000);
        vTaskDelay(1000 / portTICK_PERIOD_MS); 
    }
}
