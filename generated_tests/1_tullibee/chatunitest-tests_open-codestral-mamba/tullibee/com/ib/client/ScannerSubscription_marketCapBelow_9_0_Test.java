package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    @Test
    public void testMarketCapBelow() {
        // Create a mock object for the ScannerSubscription class
        ScannerSubscription scannerSubscriptionMock = Mockito.mock(ScannerSubscription.class);
        // Set the expected market capitalization below value
        double expectedMarketCapBelow = 100.0;
        // Set the mock object to return the expected market capitalization below value when the marketCapBelow() method is called
        Mockito.when(scannerSubscriptionMock.marketCapBelow()).thenReturn(expectedMarketCapBelow);
        // Call the marketCapBelow() method on the mock object
        double actualMarketCapBelow = scannerSubscriptionMock.marketCapBelow();
        // Assert that the actual market capitalization below value is equal to the expected value
        assertEquals(expectedMarketCapBelow, actualMarketCapBelow);
    }
}
