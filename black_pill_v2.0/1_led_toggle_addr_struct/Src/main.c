
#include <stdint.h>

#define PERIPH_BASE			(0x40000000UL)
#define AHB1_OFFSET	        (0x00020000UL)
#define AHB1		        (PERIPH_BASE + AHB1_OFFSET)
#define GPIOC_OFFSET        (0X0800UL)

#define GPIOC_BASE          (AHB1 + GPIOC_OFFSET)

#define RCC_OFFSET          (0x3800UL)
#define RCC_BASE           	(AHB1 + RCC_OFFSET)

typedef struct
{
	volatile uint32_t DUMMY[12];
	volatile uint32_t AHB1ENR;
} RCC_TypeDef;


typedef struct 
{
	volatile uint32_t MODER;
	volatile uint32_t DUMMY[4];
	volatile uint32_t ODR;
} GPIO_TypeDef;

#define RCC		((RCC_TypeDef*) RCC_BASE)
#define GPIOC	((GPIO_TypeDef*) GPIOC_BASE)

// IO port C clock enable
#define GPIOCEN         	(1U<<2)

// ODRy: Port output data (y = 0..15)
#define ODR13               (1U<<13)
#define LED_PIN             ODR13


int main(void)
{
	/*1. Enable clock access to GPIOC*/
	RCC->AHB1ENR |= GPIOCEN;

    /*2. Set PC13 as output pin*/
    GPIOC->MODER |=  (1U<<26);
    GPIOC->MODER &= ~(1U<<27);

    while(1)
    {
    	/*3. Set PC13 high*/
    	GPIOC->ODR ^= LED_PIN;

    	for (int i=0; i< 100000;i++)
    	{

    	}
    }
}


