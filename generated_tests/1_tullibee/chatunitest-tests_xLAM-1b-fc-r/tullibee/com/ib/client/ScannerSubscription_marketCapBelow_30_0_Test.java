package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    @Test
    public void marketCapBelowTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        double cap = 100000.0;
        subscription.marketCapBelow(cap);
        assertEquals(cap, subscription.marketCapBelow());
    }
}
