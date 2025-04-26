package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    void testAbovePrice() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        // Test with default value
        assertEquals(Double.MAX_VALUE, scannerSubscription.abovePrice());
        // Test with a set value
        scannerSubscription.abovePrice(100.50);
        assertEquals(100.50, scannerSubscription.abovePrice());
        // Test with zero
        scannerSubscription.abovePrice(0);
        assertEquals(0, scannerSubscription.abovePrice());
        // Test with a large value
        scannerSubscription.abovePrice(1000000.0);
        assertEquals(1000000.0, scannerSubscription.abovePrice());
        // Test with a negative value
        scannerSubscription.abovePrice(-100);
        assertEquals(-100, scannerSubscription.abovePrice());
    }
}
