package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapBelow() {
        double expectedMarketCapBelow = 1000.0;
        scannerSubscription.marketCapBelow(expectedMarketCapBelow);
        double actualMarketCapBelow = scannerSubscription.marketCapBelow();
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow, "The market cap below value should be set correctly.");
    }

    @Test
    public void testMarketCapBelowWithNegativeValue() {
        double expectedMarketCapBelow = -500.0;
        scannerSubscription.marketCapBelow(expectedMarketCapBelow);
        double actualMarketCapBelow = scannerSubscription.marketCapBelow();
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow, "The market cap below value should be set correctly even for negative values.");
    }

    @Test
    public void testMarketCapBelowWithZero() {
        double expectedMarketCapBelow = 0.0;
        scannerSubscription.marketCapBelow(expectedMarketCapBelow);
        double actualMarketCapBelow = scannerSubscription.marketCapBelow();
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow, "The market cap below value should be set correctly for zero.");
    }

    @Test
    public void testMarketCapBelowWithMaxValue() {
        double expectedMarketCapBelow = Double.MAX_VALUE;
        scannerSubscription.marketCapBelow(expectedMarketCapBelow);
        double actualMarketCapBelow = scannerSubscription.marketCapBelow();
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow, "The market cap below value should be set correctly for the maximum double value.");
    }
}
