package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_29_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapAbove() {
        double expectedMarketCapAbove = 1000.0;
        scannerSubscription.marketCapAbove(expectedMarketCapAbove);
        double actualMarketCapAbove = scannerSubscription.marketCapAbove();
        assertEquals(expectedMarketCapAbove, actualMarketCapAbove, "The market cap above should be set correctly");
    }
}
