package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    @Test
    void testMarketCapBelow() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        // Test with a valid positive value
        double validCap = 1000000.0;
        scannerSubscription.marketCapBelow(validCap);
        assertEquals(validCap, scannerSubscription.marketCapBelow());
        // Test with zero
        scannerSubscription.marketCapBelow(0.0);
        assertEquals(0.0, scannerSubscription.marketCapBelow());
        // Test with a large value
        double largeCap = Double.MAX_VALUE;
        scannerSubscription.marketCapBelow(largeCap);
        assertEquals(largeCap, scannerSubscription.marketCapBelow());
        // Test with a negative value (should still set the value)
        double negativeCap = -1000000.0;
        scannerSubscription.marketCapBelow(negativeCap);
        assertEquals(negativeCap, scannerSubscription.marketCapBelow());
        // Test with Double.NaN
        scannerSubscription.marketCapBelow(Double.NaN);
        assertEquals(Double.NaN, scannerSubscription.marketCapBelow());
        // Test with Double.POSITIVE_INFINITY
        scannerSubscription.marketCapBelow(Double.POSITIVE_INFINITY);
        assertEquals(Double.POSITIVE_INFINITY, scannerSubscription.marketCapBelow());
        // Test with Double.NEGATIVE_INFINITY
        scannerSubscription.marketCapBelow(Double.NEGATIVE_INFINITY);
        assertEquals(Double.NEGATIVE_INFINITY, scannerSubscription.marketCapBelow());
    }
}
