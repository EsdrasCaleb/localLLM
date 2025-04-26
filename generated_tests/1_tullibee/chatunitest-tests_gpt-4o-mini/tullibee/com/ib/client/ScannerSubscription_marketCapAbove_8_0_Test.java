package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapAboveDefault() {
        // Test default value
        assertEquals(Double.MAX_VALUE, scannerSubscription.marketCapAbove());
    }

    @Test
    public void testMarketCapAboveSetValue() {
        // Set a specific value and test
        double expectedValue = 1000000.0;
        scannerSubscription.marketCapAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.marketCapAbove());
    }

    @Test
    public void testMarketCapAboveNegativeValue() {
        // Set a negative value and test
        double expectedValue = -500000.0;
        scannerSubscription.marketCapAbove(expectedValue);
        assertEquals(expectedValue, scannerSubscription.marketCapAbove());
    }
}
