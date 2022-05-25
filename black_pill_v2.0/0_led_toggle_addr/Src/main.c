
#define PERIPH_BASE			(0x40000000UL)
#define AHB1_OFFSET	        (0x00020000UL)
#define AHB1		        (PERIPH_BASE + AHB1_OFFSET)
#define GPIOC_OFFSET        (0X0800UL)

#define GPIOC_BASE          (AHB1 + GPIOC_OFFSET)

#define RCC_OFFSET          (0x3800UL)
#define RCC_BASE           	(AHB1 + RCC_OFFSET)

// RCC AHB1 peripheral clock enable register (RCC_AHB1ENR)
#define RCC_AHB1ENR_OFFSET	(0x30UL)
#define RCC_AHB1ENR       	(*(volatile unsigned int *)(RCC_BASE + RCC_AHB1ENR_OFFSET))

// GPIO port mode register (GPIOx_MODER) (x = A..E and H)
#define MODER_OFFSET       	(0x00UL)
#define GPIOC_MODER        	(*(volatile unsigned int *)(GPIOC_BASE + MODER_OFFSET))

// GPIO port output data register (GPIOx_ODR) (x = A..E and H)
#define ODR_OFFSET         	(0x14UL)
#define GPIOC_ODR         	(*(volatile unsigned int *)(GPIOC_BASE + ODR_OFFSET))

// IO port C clock enable
#define GPIOCEN         	(1U<<2)

// ODRy: Port output data (y = 0..15)
#define ODR13               (1U<<13)

#define LED_PIN             ODR13


int main(void)
{
	/*1. Enable clock access to GPIOC*/
	RCC_AHB1ENR |= GPIOCEN;

    /*2. Set PC13 as output pin*/
    GPIOC_MODER |=  (1U<<26);
    GPIOC_MODER &= ~(1U<<27);

    while(1)
    {
    	/*3. Set PC13 high*/
//        GPIOC_OD_R |= LED_PIN;
    	GPIOC_ODR ^= LED_PIN;

    	for (int i=0; i< 100000;i++)
    	{

    	}
    }
}
