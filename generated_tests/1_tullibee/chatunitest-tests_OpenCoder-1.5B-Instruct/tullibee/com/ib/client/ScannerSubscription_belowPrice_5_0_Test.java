package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    public void testBelowPrice() {
        ScannerSubscription mock = Mockito.mock(ScannerSubscription.class);
        Mockito.when(mock.belowPrice()).thenReturn(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, mock.belowPrice());
    }
}
