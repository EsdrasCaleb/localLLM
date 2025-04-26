package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_marketCapAbove_29_0_Test {

    @Test
    void testMarketCapAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        double cap = 10000.50;
        subscription.marketCapAbove(cap);
        assertEquals(cap, subscription.marketCapAbove());
    }

    @Test
    void testMarketCapAbove_Zero() {
        ScannerSubscription subscription = new ScannerSubscription();
        double cap = 0;
        subscription.marketCapAbove(cap);
        assertEquals(cap, subscription.marketCapAbove());
    }

    @Test
    void testMarketCapAbove_Negative() {
        ScannerSubscription subscription = new ScannerSubscription();
        double cap = -1000.50;
        subscription.marketCapAbove(cap);
        assertEquals(cap, subscription.marketCapAbove());
    }

    @Test
    void testMarketCapAbove_MaxDouble() {
        ScannerSubscription subscription = new ScannerSubscription();
        double cap = Double.MAX_VALUE;
        subscription.marketCapAbove(cap);
        assertEquals(cap, subscription.marketCapAbove());
    }

    @Test
    void testMarketCapAbove_MinDouble() {
        ScannerSubscription subscription = new ScannerSubscription();
        double cap = Double.MIN_VALUE;
        subscription.marketCapAbove(cap);
        assertEquals(cap, subscription.marketCapAbove());
    }
}
