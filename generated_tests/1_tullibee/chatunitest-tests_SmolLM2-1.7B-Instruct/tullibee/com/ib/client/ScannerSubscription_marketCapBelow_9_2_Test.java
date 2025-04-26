package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_2_Test {

    @Test
    public void marketCapBelow_returnsMarketCapAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapAbove(100);
        subscription.marketCapBelow(50);
        assertEquals(100, subscription.marketCapBelow());
    }
}
