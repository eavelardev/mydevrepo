#include "stm32f4xx.h"

#define GPIOAEN	(1U<<0)
#define GPIOCEN	(1U<<2)

#define	PIN0	(1U<<0)
#define	PIN13	(1U<<13)

#define LED_PIN	PIN13
#define BTN_PIN	PIN0

int main(void)
{
	RCC->AHB1ENR |= (GPIOAEN | GPIOCEN);

	/* LED */
	GPIOC->MODER |=  (1U<<26);
	GPIOC->MODER &= ~(1U<<27);

	/* BTN */
	GPIOA->MODER &= ~(1U<<0 | 1U<<1);

	/* Enable pull-up */
	GPIOA->PUPDR |=  (1U<<0);
	GPIOC->PUPDR &= ~(1U<<1);

    while(1)
    {
    	if (GPIOA->IDR & BTN_PIN)
    	{
    		GPIOC->BSRR = LED_PIN;
    	}
    	else
    	{
    		GPIOC->BSRR = (1U<<29);
    	}
    }
}
