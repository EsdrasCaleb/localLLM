package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_stockTypeFilter_41_0_Test {

    @Test
    void testStockTypeFilter() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid string
        subscription.stockTypeFilter("Equity");
        Field stockTypeFilterField = ScannerSubscription.class.getDeclaredField("m_stockTypeFilter");
        stockTypeFilterField.setAccessible(true);
        String actualValue = (String) stockTypeFilterField.get(subscription);
        assertEquals("Equity", actualValue);
        // Test with null input
        subscription = new ScannerSubscription();
        subscription.stockTypeFilter(null);
        actualValue = (String) stockTypeFilterField.get(subscription);
        assertNull(actualValue);
        // Test with empty string input
        subscription = new ScannerSubscription();
        subscription.stockTypeFilter("");
        actualValue = (String) stockTypeFilterField.get(subscription);
        assertEquals("", actualValue);
    }
}
