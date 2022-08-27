#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "driver/gpio.h"

#define VIBRATOR_PIN    (gpio_num_t)26
#define BUTTON_PIN      (gpio_num_t)0

void gpio_init()
{
    gpio_pad_select_gpio(BUTTON_PIN);
    gpio_set_direction(BUTTON_PIN, GPIO_MODE_INPUT);

    gpio_pad_select_gpio(VIBRATOR_PIN);
    gpio_set_direction(VIBRATOR_PIN, GPIO_MODE_OUTPUT);
}

extern "C" void app_main()
{
    gpio_init();

    while(true)
    {
        if(gpio_get_level(BUTTON_PIN) == 0)
            gpio_set_level(VIBRATOR_PIN, 1);
        else
            gpio_set_level(VIBRATOR_PIN, 0);

        vTaskDelay(1 / portTICK_PERIOD_MS); 
    }
}
