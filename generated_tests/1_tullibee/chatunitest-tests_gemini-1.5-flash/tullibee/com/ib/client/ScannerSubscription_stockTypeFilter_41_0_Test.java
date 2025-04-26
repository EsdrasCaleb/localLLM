package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_41_0_Test {

    @Test
    void testStockTypeFilterNull() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.stockTypeFilter(null);
        assertNull(getPrivateField(scannerSubscription, "m_stockTypeFilter"));
    }

    @Test
    void testStockTypeFilterEmpty() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.stockTypeFilter("");
        assertEquals("", getPrivateField(scannerSubscription, "m_stockTypeFilter"));
    }

    @Test
    void testStockTypeFilterNonEmpty() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String testValue = "testFilter";
        scannerSubscription.stockTypeFilter(testValue);
        assertEquals(testValue, getPrivateField(scannerSubscription, "m_stockTypeFilter"));
    }

    private String getPrivateField(ScannerSubscription obj, String fieldName) {
        try {
            Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to access private field: " + e.getMessage());
            // Should not reach here due to fail()
            return null;
        }
    }
}
