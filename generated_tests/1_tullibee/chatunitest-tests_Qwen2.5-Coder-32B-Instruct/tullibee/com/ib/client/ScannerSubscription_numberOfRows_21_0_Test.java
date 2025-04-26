package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_21_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testNumberOfRows() throws NoSuchFieldException, IllegalAccessException {
        // Test with a positive number of rows
        int testRows = 10;
        scannerSubscription.numberOfRows(testRows);
        assertEquals(testRows, getPrivateFieldValue("m_numberOfRows", scannerSubscription));
        // Test with zero rows
        testRows = 0;
        scannerSubscription.numberOfRows(testRows);
        assertEquals(testRows, getPrivateFieldValue("m_numberOfRows", scannerSubscription));
        // Test with a negative number of rows
        testRows = -5;
        scannerSubscription.numberOfRows(testRows);
        assertEquals(testRows, getPrivateFieldValue("m_numberOfRows", scannerSubscription));
        // Test with the NO_ROW_NUMBER_SPECIFIED constant
        testRows = ScannerSubscription.NO_ROW_NUMBER_SPECIFIED;
        scannerSubscription.numberOfRows(testRows);
        assertEquals(testRows, getPrivateFieldValue("m_numberOfRows", scannerSubscription));
    }

    private Object getPrivateFieldValue(String fieldName, Object object) throws NoSuchFieldException, IllegalAccessException {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(object);
    }
}
