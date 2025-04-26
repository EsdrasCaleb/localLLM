package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_3_Test {

    @Test
    public void testMarketCapBelow_SetMarketCapBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapBelow(100.0);
        assertEquals(100.0, scannerSubscription.marketCapBelow(), 0.01);
    }

    @Test
    public void testMarketCapBelow_SetMarketCapBelowZero() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapBelow(0.0);
        assertEquals(0.0, scannerSubscription.marketCapBelow(), 0.01);
    }

    @Test
    public void testMarketCapBelow_SetMarketCapBelowNegative() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapBelow(-100.0);
        assertEquals(-100.0, scannerSubscription.marketCapBelow(), 0.01);
    }

    @Test
    public void testMarketCapBelow_SetMarketCapBelowMaxDouble() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapBelow(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, scannerSubscription.marketCapBelow(), 0.01);
    }

    @Test
    public void testMarketCapBelow_SetMarketCapBelowMinDouble() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapBelow(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, scannerSubscription.marketCapBelow(), 0.01);
    }

    @Test
    public void testMarketCapBelow_ThrowsException() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(ArithmeticException.class, () -> scannerSubscription.marketCapBelow(Double.POSITIVE_INFINITY));
    }
}
