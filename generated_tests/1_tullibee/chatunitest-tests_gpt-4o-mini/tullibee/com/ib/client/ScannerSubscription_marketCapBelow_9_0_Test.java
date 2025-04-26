package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapBelow_DefaultValue() {
        // Test the default value of m_marketCapBelow
        assertEquals(Double.MAX_VALUE, scannerSubscription.marketCapBelow(), "Default marketCapBelow should be Double.MAX_VALUE");
    }

    @Test
    public void testMarketCapBelow_SetValue() {
        // Set a specific value and test if marketCapBelow returns it
        double expectedValue = 1000000.0;
        scannerSubscription.marketCapBelow(expectedValue);
        assertEquals(expectedValue, scannerSubscription.marketCapBelow(), "marketCapBelow should return the value set");
    }
}
