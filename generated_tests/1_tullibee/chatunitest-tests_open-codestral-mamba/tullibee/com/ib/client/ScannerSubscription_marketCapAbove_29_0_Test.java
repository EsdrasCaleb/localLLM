package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_0_Test {

    @Test
    public void testMarketCapAbove() {
        ScannerSubscription scanner = new ScannerSubscription();
        double cap = 100.0;
        scanner.marketCapAbove(cap);
        assertEquals(cap, scanner.marketCapAbove(), "marketCapAbove method failed");
    }
}
