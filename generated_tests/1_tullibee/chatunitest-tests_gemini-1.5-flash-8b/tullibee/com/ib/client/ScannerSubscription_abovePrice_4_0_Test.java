package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    void testAbovePrice_positiveValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.abovePrice(100.50);
        assertEquals(100.50, subscription.abovePrice());
    }

    @Test
    void testAbovePrice_defaultValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Double.MAX_VALUE, subscription.abovePrice());
    }

    @Test
    void testAbovePrice_zeroValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.abovePrice(0.0);
        assertEquals(0.0, subscription.abovePrice());
    }
}
