package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    @Test
    void testMarketCapAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        // Test with default value
        assertEquals(Double.MAX_VALUE, scannerSubscription.marketCapAbove());
        // Test with a set value
        double testValue = 1000000.0;
        scannerSubscription.marketCapAbove(testValue);
        assertEquals(testValue, scannerSubscription.marketCapAbove());
        // Test with zero value
        scannerSubscription.marketCapAbove(0.0);
        assertEquals(0.0, scannerSubscription.marketCapAbove());
        // Test with a negative value
        scannerSubscription.marketCapAbove(-1000000.0);
        assertEquals(-1000000.0, scannerSubscription.marketCapAbove());
    }
}
