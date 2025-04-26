package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_2_Test {

    @Test
    public void testMarketCapBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double marketCapBelow = scannerSubscription.marketCapBelow();
        assertEquals(Double.MAX_VALUE, marketCapBelow, 0.01);
    }

    @Test
    public void testMarketCapBelow_Set() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapBelow(100.0);
        double marketCapBelow = scannerSubscription.marketCapBelow();
        assertEquals(100.0, marketCapBelow, 0.01);
    }
}
