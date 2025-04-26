package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_moodyRatingAbove_10_0_Test {

    @Test
    void moodyRatingAboveTest() {
        // Create a mock object of ScannerSubscription
        ScannerSubscription mockSubscription = Mockito.mock(ScannerSubscription.class);
        // Set the expected behavior of the mock object
        Mockito.when(mockSubscription.moodyRatingAbove()).thenReturn("Aaa");
        // Invoke the method under test
        String result = mockSubscription.moodyRatingAbove();
        // Assert the expected result
        assertEquals("Aaa", result);
    }
}
