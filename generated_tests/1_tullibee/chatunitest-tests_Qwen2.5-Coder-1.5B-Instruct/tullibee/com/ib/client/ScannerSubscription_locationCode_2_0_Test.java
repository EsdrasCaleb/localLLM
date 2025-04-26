package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    @Test
    public void testLocationCode() throws Exception {
        // Create an instance of ScannerSubscription
        ScannerSubscription scannerSubscription = mock(ScannerSubscription.class);
        // Invoke the locationCode method
        String result = scannerSubscription.locationCode();
        // Assert the result
        assertNotNull(result, "Expected a non-null location code");
        assertFalse(result.isEmpty(), "Location code should not be empty");
    }
}
