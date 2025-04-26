package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_moodyRatingBelow_11_0_Test {

    @Test
    void moodyRatingBelowTest() {
        // Create a mock of ScannerSubscription
        ScannerSubscription subscription = Mockito.mock(ScannerSubscription.class);
        // Set the behavior of the mocked method
        Mockito.when(subscription.moodyRatingBelow()).thenReturn("A");
        // Call the mocked method
        String result = subscription.moodyRatingBelow();
        // Verify the result
        assertEquals("A", result);
    }
}
