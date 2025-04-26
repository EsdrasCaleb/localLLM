package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scanCode_24_0_Test {

    @Test
    void scanCode_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String scanCode = "12345";
        subscription.scanCode(scanCode);
        assertEquals(scanCode, subscription.scanCode());
    }

    @Test
    void scanCode_nullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String scanCode = null;
        subscription.scanCode(scanCode);
        assertNull(subscription.scanCode());
    }

    @Test
    void scanCode_emptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String scanCode = "";
        subscription.scanCode(scanCode);
        assertEquals(scanCode, subscription.scanCode());
    }
}
