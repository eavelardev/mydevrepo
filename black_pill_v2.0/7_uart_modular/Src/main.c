#include <stdio.h>
#include "uart.h"

int main(void)
{
	uart2_tx_init();
	printf("Hello from STM32F4........\n\r");
}
