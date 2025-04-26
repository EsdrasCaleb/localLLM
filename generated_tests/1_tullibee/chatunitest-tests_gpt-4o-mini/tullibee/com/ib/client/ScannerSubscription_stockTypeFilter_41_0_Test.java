package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_41_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testStockTypeFilter_SetsValueCorrectly() {
        // Arrange
        String expectedValue = "Equity";
        // Act
        scannerSubscription.stockTypeFilter(expectedValue);
        // Assert
        assertEquals(expectedValue, getPrivateField(scannerSubscription, "m_stockTypeFilter"));
    }

    @Test
    public void testStockTypeFilter_SetsNullValue() {
        // Arrange
        String expectedValue = null;
        // Act
        scannerSubscription.stockTypeFilter(expectedValue);
        // Assert
        assertEquals(expectedValue, getPrivateField(scannerSubscription, "m_stockTypeFilter"));
    }

    private Object getPrivateField(ScannerSubscription scannerSubscription, String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
