/*
 * uart.h
 *
 *  Created on: Mar 7, 2022
 *      Author: eduar
 */

#ifndef UART_H_
#define UART_H_
#include "stm32f4xx.h"

void uart2_rttx_init(void);
char uart2_read(void);
void uart2_write(int ch);

#endif /* UART_H_ */
