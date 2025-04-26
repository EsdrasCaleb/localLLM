package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_1_Test {

    @Test
    void marketCapAbove() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.marketCapAbove()).thenReturn(10.0);
        assertEquals(10.0, scannerSubscription.marketCapAbove());
    }
}
