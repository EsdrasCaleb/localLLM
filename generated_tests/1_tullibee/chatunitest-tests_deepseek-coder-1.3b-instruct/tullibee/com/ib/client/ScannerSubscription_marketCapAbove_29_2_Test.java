package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_2_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setup() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapAbove() {
        double testValue = 123.45;
        scannerSubscription.marketCapAbove(testValue);
        assertEquals(testValue, scannerSubscription.marketCapAbove());
    }
}
