package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    @Test
    void testMarketCapBelow_Default() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Double.MAX_VALUE, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_SetAndGet() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedMarketCap = 1000.0;
        subscription.marketCapBelow(expectedMarketCap);
        assertEquals(expectedMarketCap, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_Zero() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapBelow(0.0);
        assertEquals(0.0, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_Negative() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapBelow(-100.0);
        assertEquals(-100.0, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_LargeValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapBelow(1e12);
        assertEquals(1e12, subscription.marketCapBelow());
    }
}
