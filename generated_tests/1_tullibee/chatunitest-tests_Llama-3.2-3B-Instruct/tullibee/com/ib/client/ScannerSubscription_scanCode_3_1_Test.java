package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_1_Test {

    @Test
    public void testScanCode_Getter() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.instrument("Instrument1");
        subscription.locationCode("Location1");
        subscription.scanCode("ScanCode1");
        assertEquals("ScanCode1", subscription.scanCode());
    }

    @Test
    public void testScanCode_Null() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> subscription.scanCode());
    }

    @Test
    public void testScanCode_Empty() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals("", subscription.scanCode());
    }
}
