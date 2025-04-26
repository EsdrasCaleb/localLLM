package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_0_Test {

    @Test
    void testScanCode_NoScanCodeSet() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertNull(subscription.scanCode(), "Scan code should be null when not set");
    }

    @Test
    void testScanCode_ScanCodeSet() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedScanCode = "12345";
        subscription.scanCode(expectedScanCode);
        assertEquals(expectedScanCode, subscription.scanCode(), "Scan code should match the set value");
    }

    @Test
    void testScanCode_ScanCodeSetThenCleared() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scanCode("12345");
        subscription.scanCode(null);
        assertNull(subscription.scanCode(), "Scan code should be null after clearing");
    }

    @Test
    void testScanCode_ScanCodeSetToEmpty() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scanCode("");
        assertEquals("", subscription.scanCode(), "Scan code should be empty string when set to empty");
    }
}
