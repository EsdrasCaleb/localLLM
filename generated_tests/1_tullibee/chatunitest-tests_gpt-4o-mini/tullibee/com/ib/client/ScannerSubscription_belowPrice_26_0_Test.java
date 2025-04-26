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
        // Test with a normal value
        double price = 100.50;
        scannerSubscription.belowPrice(price);
        assertEquals(price, scannerSubscription.belowPrice(), "The belowPrice should be set correctly.");
        // Test with a negative value
        double negativePrice = -50.0;
        scannerSubscription.belowPrice(negativePrice);
        assertEquals(negativePrice, scannerSubscription.belowPrice(), "The belowPrice should accept negative values.");
        // Test with zero
        double zeroPrice = 0.0;
        scannerSubscription.belowPrice(zeroPrice);
        assertEquals(zeroPrice, scannerSubscription.belowPrice(), "The belowPrice should accept zero as a valid price.");
        // Test with Double.MAX_VALUE
        double maxValue = Double.MAX_VALUE;
        scannerSubscription.belowPrice(maxValue);
        assertEquals(maxValue, scannerSubscription.belowPrice(), "The belowPrice should accept Double.MAX_VALUE.");
        // Test with Double.MIN_VALUE
        double minValue = Double.MIN_VALUE;
        scannerSubscription.belowPrice(minValue);
        assertEquals(minValue, scannerSubscription.belowPrice(), "The belowPrice should accept Double.MIN_VALUE.");
    }
}
