#include <stdio.h>
#include "uart.h"

char key;

int main(void)
{
	uart2_rttx_init();

	while(1)
	{
		key = uart2_read();
		uart2_write(key);
	}
}
