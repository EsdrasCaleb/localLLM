package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_0_Test {

    @Test
    public void testScanCode() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expected = "Expected scan code";
        subscription.scanCode(expected);
        assertEquals(expected, subscription.scanCode());
    }
}
