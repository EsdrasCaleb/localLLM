package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_marketCapAbove_29_0_Test {

    @Test
    void marketCapAboveTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        double marketCap = 1000.0;
        subscription.marketCapAbove(marketCap);
        assertEquals(marketCap, subscription.marketCapAbove());
    }
}
