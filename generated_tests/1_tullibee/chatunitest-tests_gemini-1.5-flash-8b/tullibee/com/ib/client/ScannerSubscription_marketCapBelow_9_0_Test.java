package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_marketCapBelow_9_0_Test {

    @Test
    void testMarketCapBelow_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double marketCap = 1000.50;
        subscription.marketCapBelow(marketCap);
        assertEquals(marketCap, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_initialDefault() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Double.MAX_VALUE, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_zeroInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapBelow(0);
        assertEquals(0, subscription.marketCapBelow());
    }

    @Test
    void testMarketCapBelow_negativeInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapBelow(-100.25);
        assertEquals(-100.25, subscription.marketCapBelow());
    }
}
