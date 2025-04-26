package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    public void testAbovePrice() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.abovePrice()).thenReturn(100.0);
        double result = scannerSubscription.abovePrice();
        assertEquals(100.0, result);
    }
}
