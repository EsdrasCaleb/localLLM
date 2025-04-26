package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_1_Test {

    @Test
    public void testMarketCapAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        double marketCap = 1000.0;
        subscription.marketCapAbove(marketCap);
        assertEquals(marketCap, subscription.marketCapAbove());
    }
}
