package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_20_2_Test {

    @Test
    public void testStockTypeFilter() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        when(scannerSubscription.stockTypeFilter()).thenReturn("A");
        String result = scannerSubscription.stockTypeFilter();
        assertEquals("A", result);
    }
}
