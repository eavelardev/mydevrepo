#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "LIS3DHTR.h"

/** Number of axis used and sample data format */
#define INERTIAL_AXIS_SAMPLED   3

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f

static float imu_data[INERTIAL_AXIS_SAMPLED];

LIS3DHTR lis;

bool inertial_init(void) {

    lis.begin(LIS3DHTR_ADDRESS_UPDATED);

    if(lis.isConnection() == false) {
        printf("ERR: failed to connect to inertial sensor!\n");
        return false;
    }

    vTaskDelay(100 / portTICK_RATE_MS);
    lis.setFullScaleRange(LIS3DHTR_RANGE_2G);
    lis.setOutputDataRate(LIS3DHTR_DATARATE_100HZ);

    return true;
}

extern "C" void app_main()
{
    /* Setup the inertial sensor */
    if (inertial_init() == false) {
        printf("Inertial sensor initialization failed\r\n");
        return;
    }

    for (int i=0; i<10; i++)
    {
        lis.getAcceleration(&imu_data[0], &imu_data[1], &imu_data[2]);
        imu_data[0] *= CONVERT_G_TO_MS2;
        imu_data[1] *= CONVERT_G_TO_MS2;
        imu_data[2] *= CONVERT_G_TO_MS2;

        printf("accX: %f\t accY: %f\t accZ: %f\n", imu_data[0], imu_data[1], imu_data[2]);
        vTaskDelay(1000 / portTICK_RATE_MS);
    }

    printf("Finish!\n");
}
