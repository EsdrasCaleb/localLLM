package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_1_Test {

    @Test
    public void marketCapAboveTest() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double expectedMarketCapAbove = 1000000.0;
        scannerSubscription.marketCapAbove(expectedMarketCapAbove);
        assertEquals(expectedMarketCapAbove, scannerSubscription.marketCapAbove());
    }
}
