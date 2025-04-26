package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_30_0_Test {

    @Test
    public void testMarketCapBelow() {
        // Create a mock object of ScannerSubscription
        ScannerSubscription subscription = Mockito.mock(ScannerSubscription.class);
        // Set the expected market capitalization value
        double expectedCap = 100.0;
        // Call the method under test
        subscription.marketCapBelow(expectedCap);
        // Verify that the m_marketCapBelow field is set correctly
        assertEquals(expectedCap, subscription.marketCapBelow());
    }
}
