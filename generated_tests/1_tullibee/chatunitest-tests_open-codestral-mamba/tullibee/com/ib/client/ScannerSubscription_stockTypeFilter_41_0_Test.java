package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_stockTypeFilter_41_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void testStockTypeFilter() {
        scannerSubscription.stockTypeFilter("Common Stock");
        assertEquals("Common Stock", scannerSubscription.stockTypeFilter());
        scannerSubscription.stockTypeFilter("");
        assertEquals("", scannerSubscription.stockTypeFilter());
        scannerSubscription.stockTypeFilter(null);
        assertNull(scannerSubscription.stockTypeFilter());
    }
}
