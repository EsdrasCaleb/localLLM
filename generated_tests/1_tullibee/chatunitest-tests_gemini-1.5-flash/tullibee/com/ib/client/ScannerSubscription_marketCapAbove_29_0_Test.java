package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_0_Test {

    @Test
    void testMarketCapAbove() throws Exception {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid positive value
        double validCap = 1000000.0;
        subscription.marketCapAbove(validCap);
        assertEquals(validCap, subscription.marketCapAbove());
        // Test with zero
        subscription.marketCapAbove(0.0);
        assertEquals(0.0, subscription.marketCapAbove());
        // Test with a large value
        double largeCap = Double.MAX_VALUE;
        subscription.marketCapAbove(largeCap);
        assertEquals(largeCap, subscription.marketCapAbove());
        // Test with a negative value (should still set the value)
        double negativeCap = -1000000.0;
        subscription.marketCapAbove(negativeCap);
        assertEquals(negativeCap, subscription.marketCapAbove());
    }
}
