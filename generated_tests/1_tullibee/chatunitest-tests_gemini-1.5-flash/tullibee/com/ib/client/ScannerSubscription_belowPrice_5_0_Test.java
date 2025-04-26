package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    void testBelowPrice() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        // Test with default value
        assertEquals(Double.MAX_VALUE, scannerSubscription.belowPrice());
        // Test with a set value
        double testPrice = 123.45;
        scannerSubscription.belowPrice(testPrice);
        assertEquals(testPrice, scannerSubscription.belowPrice());
        // Test with another set value
        double anotherTestPrice = 678.90;
        scannerSubscription.belowPrice(anotherTestPrice);
        assertEquals(anotherTestPrice, scannerSubscription.belowPrice());
        // Test with zero
        scannerSubscription.belowPrice(0);
        assertEquals(0, scannerSubscription.belowPrice());
        // Test with negative value
        scannerSubscription.belowPrice(-100);
        assertEquals(-100, scannerSubscription.belowPrice());
    }
}
