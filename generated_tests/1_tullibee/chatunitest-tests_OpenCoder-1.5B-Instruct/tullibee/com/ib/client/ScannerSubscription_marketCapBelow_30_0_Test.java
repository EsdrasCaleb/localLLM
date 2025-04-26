package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    @Test
    public void testMarketCapBelow() {
        ScannerSubscription scannerSubscription = Mockito.spy(new ScannerSubscription());
        when(scannerSubscription.marketCapBelow()).thenReturn(Double.MAX_VALUE);
        scannerSubscription.marketCapBelow(100.0);
        assertEquals(100.0, scannerSubscription.marketCapBelow());
    }
}
