package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_20_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testStockTypeFilter_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Access the private field m_stockTypeFilter using reflection
        Field field = ScannerSubscription.class.getDeclaredField("m_stockTypeFilter");
        field.setAccessible(true);
        // Ensure the default value is null
        assertNull(field.get(scannerSubscription));
        // Test the stockTypeFilter method
        assertNull(scannerSubscription.stockTypeFilter());
    }

    @Test
    public void testStockTypeFilter_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a value to m_stockTypeFilter using reflection
        Field field = ScannerSubscription.class.getDeclaredField("m_stockTypeFilter");
        field.setAccessible(true);
        String testValue = "STK";
        field.set(scannerSubscription, testValue);
        // Test the stockTypeFilter method
        assertEquals(testValue, scannerSubscription.stockTypeFilter());
    }
}
