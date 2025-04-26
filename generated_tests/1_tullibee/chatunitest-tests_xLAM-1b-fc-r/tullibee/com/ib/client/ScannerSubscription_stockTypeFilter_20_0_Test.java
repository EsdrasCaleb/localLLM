package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_20_0_Test {

    @Test
    public void testStockTypeFilter() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.stockTypeFilter("TEST");
        assertEquals("TEST", subscription.stockTypeFilter());
    }
}
