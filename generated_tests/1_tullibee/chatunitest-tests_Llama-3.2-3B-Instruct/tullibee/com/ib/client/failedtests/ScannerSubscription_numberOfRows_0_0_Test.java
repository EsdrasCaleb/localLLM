package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    public void testNumberOfRows() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        // Act
        int rows = subscription.numberOfRows();
        // Assert
        assertEquals(10, rows);
    }

    @Test
    public void testNumberOfRowsDefault() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        // Act
        int rows = subscription.numberOfRows();
        // Assert
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, rows);
    }

    @Test
    public void testNumberOfRowsThrowWhenSet() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        // Act and Assert
        assertThrows(NullPointerException.class, () -> subscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsThrowWhenGet() throws Exception {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        // Act and Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_numberOfRows");
        field.setAccessible(true);
        field.set(subscription, null);
        int rows = subscription.numberOfRows();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, rows);
    }
}
