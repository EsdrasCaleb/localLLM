package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_1_Test {

    @Test
    void scanCode() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.scanCode()).thenReturn("TEST");
        assertEquals("TEST", scannerSubscription.scanCode());
    }
}
