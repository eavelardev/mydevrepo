#include "driver/gpio.h"

#define VDD_PER_ON  (gpio_num_t)4

void gpio_init()
{
    gpio_pad_select_gpio(VDD_PER_ON);
    gpio_set_direction(VDD_PER_ON, GPIO_MODE_OUTPUT);
}

void disable_devices()
{
    // disable leds, buzzer and vibrator
    gpio_set_level(VDD_PER_ON, 1);
}

extern "C" int app_main()
{
    gpio_init();
    disable_devices();

    return 0;
}
