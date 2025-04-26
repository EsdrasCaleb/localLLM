package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    @Test
    public void testMarketCapBelow() {
        ScannerSubscription subscription = mock(ScannerSubscription.class);
        when(subscription.marketCapBelow()).thenReturn(123.45);
        double result = subscription.marketCapBelow();
        assertEquals(123.45, result);
    }
}
