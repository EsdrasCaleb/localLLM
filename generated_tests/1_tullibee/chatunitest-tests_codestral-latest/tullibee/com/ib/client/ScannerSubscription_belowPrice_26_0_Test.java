package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_26_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testBelowPrice() {
        double price = 100.50;
        scannerSubscription.belowPrice(price);
        assertEquals(price, scannerSubscription.belowPrice(), "The below price should be set correctly");
    }

    @Test
    public void testBelowPriceWithZero() {
        double price = 0.0;
        scannerSubscription.belowPrice(price);
        assertEquals(price, scannerSubscription.belowPrice(), "The below price should be set to zero");
    }

    @Test
    public void testBelowPriceWithMaxValue() {
        double price = Double.MAX_VALUE;
        scannerSubscription.belowPrice(price);
        assertEquals(price, scannerSubscription.belowPrice(), "The below price should be set to Double.MAX_VALUE");
    }

    @Test
    public void testBelowPriceWithNegativeValue() {
        double price = -100.50;
        scannerSubscription.belowPrice(price);
        assertEquals(price, scannerSubscription.belowPrice(), "The below price should be set to a negative value");
    }
}
