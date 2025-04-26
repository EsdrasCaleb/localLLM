package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_3_Test {

    @Test
    public void testScanCode() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scanCode("ABC123");
        assertEquals("ABC123", subscription.scanCode());
    }
}
