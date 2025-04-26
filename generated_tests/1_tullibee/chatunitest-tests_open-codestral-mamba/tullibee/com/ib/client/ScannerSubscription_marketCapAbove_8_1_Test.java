package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_1_Test {

    @Test
    public void testMarketCapAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapAbove(100.0);
        assertEquals(100.0, scannerSubscription.marketCapAbove());
    }
}
