package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    @Test
    public void testMarketCapAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double expectedMarketCapAbove = 1000000000000.0;
        scannerSubscription.marketCapAbove(expectedMarketCapAbove);
        double actualMarketCapAbove = scannerSubscription.marketCapAbove();
        assertEquals(expectedMarketCapAbove, actualMarketCapAbove, 0.001);
    }
}
