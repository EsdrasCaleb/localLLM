package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapBelow() {
        // Test with a standard value
        double testValue1 = 1000000.0;
        scannerSubscription.marketCapBelow(testValue1);
        assertEquals(testValue1, scannerSubscription.marketCapBelow(), "Market cap below should be set correctly.");
        // Test with a negative value
        double testValue2 = -500000.0;
        scannerSubscription.marketCapBelow(testValue2);
        assertEquals(testValue2, scannerSubscription.marketCapBelow(), "Market cap below should handle negative values.");
        // Test with zero
        double testValue3 = 0.0;
        scannerSubscription.marketCapBelow(testValue3);
        assertEquals(testValue3, scannerSubscription.marketCapBelow(), "Market cap below should handle zero.");
        // Test with a very large value
        double testValue4 = Double.MAX_VALUE;
        scannerSubscription.marketCapBelow(testValue4);
        assertEquals(testValue4, scannerSubscription.marketCapBelow(), "Market cap below should handle maximum double value.");
    }
}
