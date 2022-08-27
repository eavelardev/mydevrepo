#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "driver/gpio.h"

#define VIBRATOR_PIN  (gpio_num_t)26

void gpio_init()
{
    gpio_pad_select_gpio(VIBRATOR_PIN);
    gpio_set_direction(VIBRATOR_PIN, GPIO_MODE_OUTPUT);
}

extern "C" void app_main()
{
    gpio_init();

    for (int i = 0; i < 3; i++)
    {
        gpio_set_level(VIBRATOR_PIN, 1);
        vTaskDelay(1000 / portTICK_PERIOD_MS); 
        gpio_set_level(VIBRATOR_PIN, 0);
        vTaskDelay(1000 / portTICK_PERIOD_MS); 
    }
}
