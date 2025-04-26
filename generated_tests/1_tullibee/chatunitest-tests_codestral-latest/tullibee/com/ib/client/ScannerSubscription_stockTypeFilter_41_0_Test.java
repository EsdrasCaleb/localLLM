package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_stockTypeFilter_41_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testStockTypeFilter() throws NoSuchFieldException, IllegalAccessException {
        // Test setting a non-null value
        scannerSubscription.stockTypeFilter("Common Stock");
        Field stockTypeFilterField = ScannerSubscription.class.getDeclaredField("m_stockTypeFilter");
        stockTypeFilterField.setAccessible(true);
        assertEquals("Common Stock", stockTypeFilterField.get(scannerSubscription));
        // Test setting a null value
        scannerSubscription.stockTypeFilter(null);
        assertNull(stockTypeFilterField.get(scannerSubscription));
    }
}
