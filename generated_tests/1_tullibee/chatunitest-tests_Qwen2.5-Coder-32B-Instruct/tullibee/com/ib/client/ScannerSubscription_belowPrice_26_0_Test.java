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
    public void testBelowPrice() throws Exception {
        // Test with a positive price
        double testPrice = 100.50;
        scannerSubscription.belowPrice(testPrice);
        assertEquals(testPrice, scannerSubscription.belowPrice(), "The belowPrice should be set to 100.50");
        // Test with zero price
        testPrice = 0.0;
        scannerSubscription.belowPrice(testPrice);
        assertEquals(testPrice, scannerSubscription.belowPrice(), "The belowPrice should be set to 0.0");
        // Test with a negative price
        testPrice = -50.0;
        scannerSubscription.belowPrice(testPrice);
        assertEquals(testPrice, scannerSubscription.belowPrice(), "The belowPrice should be set to -50.0");
        // Test with Double.MAX_VALUE
        testPrice = Double.MAX_VALUE;
        scannerSubscription.belowPrice(testPrice);
        assertEquals(testPrice, scannerSubscription.belowPrice(), "The belowPrice should be set to Double.MAX_VALUE");
        // Test with Double.MIN_VALUE
        testPrice = Double.MIN_VALUE;
        scannerSubscription.belowPrice(testPrice);
        assertEquals(testPrice, scannerSubscription.belowPrice(), "The belowPrice should be set to Double.MIN_VALUE");
    }
}
