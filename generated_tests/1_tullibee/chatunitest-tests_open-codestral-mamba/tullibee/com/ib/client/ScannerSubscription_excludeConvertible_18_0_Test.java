package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    // Test method for ScannerSubscription.excludeConvertible()
    @Test
    public void testExcludeConvertible() {
        // Create a mock object of ScannerSubscription
        ScannerSubscription scannerMock = Mockito.mock(ScannerSubscription.class);
        // Set up the mock to return a specific value when excludeConvertible() is called
        Mockito.when(scannerMock.excludeConvertible()).thenReturn("Yes");
        // Test the return value of excludeConvertible()
        assertEquals("Yes", scannerMock.excludeConvertible());
    }
}
