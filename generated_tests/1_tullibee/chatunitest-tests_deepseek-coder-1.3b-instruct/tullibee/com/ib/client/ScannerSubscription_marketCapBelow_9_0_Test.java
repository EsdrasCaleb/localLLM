package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_marketCapBelow_9_0_Test {

    @Test
    void testMarketCapBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedMarketCapBelow = 1000000.0;
        subscription.marketCapBelow(expectedMarketCapBelow);
        double actualMarketCapBelow = subscription.marketCapBelow();
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow);
    }
}
