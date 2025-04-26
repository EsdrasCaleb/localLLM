package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_0_Test {

    @Test
    void testMarketCapAbove() {
        ScannerSubscription instance = new ScannerSubscription();
        instance.marketCapAbove(100.0);
        assertEquals(100.0, instance.marketCapAbove());
    }
}
