package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scanCode_3_0_Test {

    @Test
    void testScanCode() {
        // Create a new ScannerSubscription object
        ScannerSubscription subscription = new ScannerSubscription();
        // Set a scan code for the subscription
        subscription.scanCode("12345");
        // Verify that the scan code is correctly retrieved
        assertEquals("12345", subscription.scanCode());
    }
}
