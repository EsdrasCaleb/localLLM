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
        // Positive case:  Valid scan code
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scanCode("12345");
        assertEquals("12345", subscription.scanCode());
        // Negative case:  Empty scan code
        ScannerSubscription subscription2 = new ScannerSubscription();
        subscription2.scanCode("");
        assertEquals("", subscription2.scanCode());
        // Negative case: Null scan code
        ScannerSubscription subscription3 = new ScannerSubscription();
        subscription3.scanCode(null);
        assertNull(subscription3.scanCode());
    }
}
