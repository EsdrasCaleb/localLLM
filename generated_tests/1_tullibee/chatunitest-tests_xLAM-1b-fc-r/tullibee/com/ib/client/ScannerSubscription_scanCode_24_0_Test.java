package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scanCode_24_0_Test {

    @Test
    void testScanCode() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scanCode("test");
        assertEquals("test", subscription.scanCode());
    }
}
