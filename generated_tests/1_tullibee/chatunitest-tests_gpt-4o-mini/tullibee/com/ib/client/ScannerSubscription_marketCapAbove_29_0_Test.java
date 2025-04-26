package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapAbove() {
        // Test with a normal value
        double testCap = 1000000.0;
        scannerSubscription.marketCapAbove(testCap);
        assertEquals(testCap, scannerSubscription.marketCapAbove());
        // Test with zero
        testCap = 0.0;
        scannerSubscription.marketCapAbove(testCap);
        assertEquals(testCap, scannerSubscription.marketCapAbove());
        // Test with a negative value
        testCap = -500000.0;
        scannerSubscription.marketCapAbove(testCap);
        assertEquals(testCap, scannerSubscription.marketCapAbove());
        // Test with Double.MAX_VALUE
        testCap = Double.MAX_VALUE;
        scannerSubscription.marketCapAbove(testCap);
        assertEquals(testCap, scannerSubscription.marketCapAbove());
        // Test with Double.MIN_VALUE
        testCap = Double.MIN_VALUE;
        scannerSubscription.marketCapAbove(testCap);
        assertEquals(testCap, scannerSubscription.marketCapAbove());
    }
}
