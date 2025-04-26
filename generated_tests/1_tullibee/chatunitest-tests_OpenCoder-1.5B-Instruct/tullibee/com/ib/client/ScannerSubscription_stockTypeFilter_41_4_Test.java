package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_41_4_Test {

    public static class ScannerSubscription {

        private String m_stockTypeFilter;

        public void stockTypeFilter(String val) {
            m_stockTypeFilter = val;
        }

        public String getStockTypeFilter() {
            return m_stockTypeFilter;
        }
    }

    @Test
    public void testStockTypeFilter() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        Method method = scannerSubscription.getClass().getDeclaredMethod("stockTypeFilter", String.class);
        method.setAccessible(true);
        // Test case 1: No filter
        method.invoke(scannerSubscription, "ALL");
        assertEquals("ALL", scannerSubscription.getStockTypeFilter());
        // Test case 2: Convertible filter
        method.invoke(scannerSubscription, "CONVERTIBLE");
        assertEquals("CONVERTIBLE", scannerSubscription.getStockTypeFilter());
        // Test case 3: Non-convertible filter
        method.invoke(scannerSubscription, "NON_CONVERTIBLE");
        assertEquals("NON_CONVERTIBLE", scannerSubscription.getStockTypeFilter());
    }
}
