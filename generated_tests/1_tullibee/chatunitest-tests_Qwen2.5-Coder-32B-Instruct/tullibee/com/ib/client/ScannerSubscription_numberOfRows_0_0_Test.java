package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testNumberOfRows_DefaultValue() {
        // Default value should be NO_ROW_NUMBER_SPECIFIED (-1)
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a specific number of rows using reflection
        Field numberOfRowsField = ScannerSubscription.class.getDeclaredField("m_numberOfRows");
        numberOfRowsField.setAccessible(true);
        numberOfRowsField.set(scannerSubscription, 100);
        // Verify that the numberOfRows() method returns the set value
        assertEquals(100, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_SetZero() throws NoSuchFieldException, IllegalAccessException {
        // Set zero rows using reflection
        Field numberOfRowsField = ScannerSubscription.class.getDeclaredField("m_numberOfRows");
        numberOfRowsField.setAccessible(true);
        numberOfRowsField.set(scannerSubscription, 0);
        // Verify that the numberOfRows() method returns zero
        assertEquals(0, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_SetNegativeValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a negative number of rows using reflection
        Field numberOfRowsField = ScannerSubscription.class.getDeclaredField("m_numberOfRows");
        numberOfRowsField.setAccessible(true);
        numberOfRowsField.set(scannerSubscription, -50);
        // Verify that the numberOfRows() method returns the set negative value
        assertEquals(-50, scannerSubscription.numberOfRows());
    }
}
