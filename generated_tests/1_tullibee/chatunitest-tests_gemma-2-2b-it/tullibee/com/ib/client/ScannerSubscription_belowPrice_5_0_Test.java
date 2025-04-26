package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    void belowPrice() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.belowPrice()).thenReturn(100.0);
        assertEquals(100.0, scannerSubscription.belowPrice());
    }
}
